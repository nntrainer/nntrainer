#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream LFM2 MoE BF16 weights into the shared Gemma4 HTP record layouts.

Experts: exact ternary -> native TL2CX_WH, gate/up fused per expert.
FC/head/conv projections: original affine INT4 grids -> QS4CX_WH_AFFINE.
Embedding, short-conv, norms and router: losslessly promoted to FP32.
A positional file is paired with a complete named offset manifest. Every
serialized expert and FC is decoded and checked before publishing the pack.
Requires numpy; the native TL2 library is built from this tree with a host C
compiler. No Transformers import, intermediate FP32 checkpoint or network.
"""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np
from convert_gemma4_fc_wh import bf16_round, pack, recover, verify_chunk, sha256


class Checkpoint:
    def __init__(self, source):
        self.path = source / 'model.safetensors'
        self.file = self.path.open('rb')
        self.stamp = self.path.stat()
        raw = self.file.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors length')
        length, = struct.unpack('<Q', raw)
        if length > 16 * 1024**2:
            raise ValueError('Unexpected safetensors header size')
        encoded = self.file.read(length)
        self.header_hash = hashlib.sha256(raw + encoded).hexdigest()
        self.header = json.loads(encoded)
        self.header.pop('__metadata__', None)
        self.start = 8 + length
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        extents = []
        for name, t in self.header.items():
            if t['dtype'] not in ('BF16', 'F32'):
                raise ValueError(f'Unsupported source dtype: {name}')
            a, b = t['data_offsets']
            expected = math.prod(t['shape']) * (2 if t['dtype'] == 'BF16' else 4)
            if b - a != expected or a < 0 or self.start + b > len(self.mm):
                raise ValueError(f'Invalid tensor extent: {name}')
            extents.append((a, b))
        end = 0
        for a, b in sorted(extents):
            if a != end:
                raise ValueError('Non-contiguous safetensors data')
            end = b
        if self.start + end != len(self.mm):
            raise ValueError('Unexpected safetensors length')

    def read(self, name, first=None, last=None):
        t = self.header[name]
        dtype = '<u2' if t['dtype'] == 'BF16' else '<f4'
        values = np.frombuffer(self.mm, dtype, math.prod(t['shape']),
                               self.start + t['data_offsets'][0]).reshape(t['shape'])
        if first is not None:
            values = values[first:last]
        if t['dtype'] == 'BF16':
            values = (values.astype(np.uint32) << 16).view(np.float32)
        else:
            values = values.copy()
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite source: {name}')
        return values

    def close(self):
        self.mm.close()
        self.file.close()


def validate_config(cfg):
    if cfg.get('model_type') != 'lfm2_moe' or cfg.get('tie_word_embeddings', True):
        raise ValueError('Expected untied LFM2 MoE')
    h, heads = cfg['hidden_size'], cfg['num_attention_heads']
    if h % heads or h // heads != 64 or heads % cfg['num_key_value_heads']:
        raise ValueError('HTP LFM2 requires head_dim=64 and complete GQA groups')
    if not 0 < cfg['num_experts'] <= 64 or cfg['num_experts'] % 32 or not 0 < cfg['num_experts_per_tok'] <= cfg['num_experts']:
        raise ValueError('Invalid expert geometry')
    if cfg.get('conv_L_cache') != 3 or cfg.get('conv_bias', False):
        raise ValueError('Only bias-free three-tap short convolution is supported')
    if not cfg.get('norm_topk_prob', True) or cfg.get('routed_scaling_factor', 1) != 1:
        raise ValueError('Unsupported router normalization')
    if not cfg.get('use_expert_bias', True):
        raise ValueError('Expected expert bias')
    if len(cfg['layer_types']) != cfg['num_hidden_layers'] or not 0 <= cfg['num_dense_layers'] < cfg['num_hidden_layers']:
        raise ValueError('Invalid layer counts')
    if cfg['num_hidden_layers'] > 64 or any(t not in ('conv', 'full_attention') for t in cfg['layer_types']):
        raise ValueError('Unsupported layer layout')
    dense = cfg['intermediate_size']
    width = next((w for w in range(min(dense, 1792), 31, -32) if dense % w == 0), 0)
    if dense % 64 or not width or dense // width > 16:
        raise ValueError('Dense FFN exceeds the WH intermediate chunk limit')
    for x in (h, cfg['moe_intermediate_size']):
        if x % 128 or not 0 < x <= 8192:
            raise ValueError('TL2 requires complete 128-channel panels')


def weight_plan(cfg, checkpoint):
    """Match the fused layers' requestWeight order, including folded norms."""
    validate_config(cfg)
    plan, used = [], set()
    h, dense, inter = cfg['hidden_size'], cfg['num_dense_layers'], cfg['moe_intermediate_size']
    kv = cfg['num_key_value_heads'] * 64
    cursor = 0

    def add(name, source, shape, dtype='FP32', transform='identity'):
        nonlocal cursor
        names = source if isinstance(source, list) else [source]
        for key in names:
            if key in used:
                raise ValueError(f'Duplicate source tensor: {key}')
            used.add(key)
        if dtype in ('QS4CX_WH_AFFINE', 'TL2CX_WH'):
            k, n = shape
            expected = [[n // len(names), k]] * len(names)
            if dtype == 'TL2CX_WH':
                size = ((k + 23) // 24) * ((n + 127) // 128) * 640 + 8 * n
            else:
                if k % 32 or n % 32:
                    raise ValueError('Affine WH requires complete 32x32 tiles')
                size = k * n // 2 + 12 * n
        else:
            size = math.prod(shape) * 4
            if transform == 'transpose': expected = [shape[::-1]]
            elif transform == 'conv_reverse': expected = [[h, 1, 3]]
            else: expected = [shape]
        for key, want in zip(names, expected):
            if checkpoint.header[key]['shape'] != want:
                raise ValueError(f'Shape mismatch: {key}: expected {want}')
        plan.append(dict(name=name, source=names, shape=shape, dtype=dtype,
                         transform=transform, offset=cursor, bytes=size))
        cursor += size

    add('embedding0:Embedding', 'model.embed_tokens.weight', [cfg['vocab_size'], h])
    for l, kind in enumerate(cfg['layer_types']):
        p, src = f'layer{l}', f'model.layers.{l}.'
        if kind == 'conv':
            add(p + '_conv_norm:gamma', src + 'operator_norm.weight', [h])
            add(p + '_conv:in_proj', src + 'conv.in_proj.weight', [h, 3*h], 'QS4CX_WH_AFFINE')
            add(p + '_conv:conv', src + 'conv.conv.weight', [3, h], transform='conv_reverse')
            add(p + '_conv:out_proj', src + 'conv.out_proj.weight', [h, h], 'QS4CX_WH_AFFINE')
        else:
            add(p + '_qkv:in_norm_gamma', src + 'operator_norm.weight', [h])
            for kind, n in [('q', h), ('k', kv), ('v', kv)]:
                add(p + f'_qkv:{kind}weight', src + f'self_attn.{kind}_proj.weight', [h, n], 'QS4CX_WH_AFFINE')
                if kind != 'v':
                    add(p + f'_qkv:{kind}_norm_gamma', src + f'self_attn.{kind}_layernorm.weight', [64])
            add(p + '_attention_out:weight', src + 'self_attn.out_proj.weight', [h, h], 'QS4CX_WH_AFFINE')
        if l < dense:
            add(p + '_ffn:in_norm_gamma', src + 'ffn_norm.weight', [h])
            for target, original, shape in [('gate', 'w1', [h, cfg['intermediate_size']]),
                                             ('up', 'w3', [h, cfg['intermediate_size']]),
                                             ('down', 'w2', [cfg['intermediate_size'], h])]:
                add(p + '_ffn:' + target, src + f'feed_forward.{original}.weight', shape, 'QS4CX_WH_AFFINE')
        else:
            m = p + '_sparse_moe:'
            add(m + 'in_norm_gamma', src + 'ffn_norm.weight', [h])
            add(m + 'gate', src + 'feed_forward.gate.weight', [h, cfg['num_experts']], transform='transpose')
            add(m + 'expert_bias', src + 'feed_forward.expert_bias', [cfg['num_experts']])
            for e in range(cfg['num_experts']):
                ex = src + f'feed_forward.experts.{e}.'
                add(m + f'expert_gate_up_{e}', [ex + 'w1.weight', ex + 'w3.weight'], [h, 2*inter], 'TL2CX_WH')
                add(m + f'expert_down_{e}', ex + 'w2.weight', [inter, h], 'TL2CX_WH')
    add('output_norm:gamma', 'model.embedding_norm.weight', [h])
    add('output_of_causallm:weight', 'lm_head.weight', [h, cfg['vocab_size']], 'QS4CX_WH_AFFINE')
    if used != checkpoint.header.keys():
        raise ValueError(f'Unaccounted source tensors: {sorted(checkpoint.header.keys() - used)}')
    return plan, cursor


class Tl2:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        for name in ('hvx_tl2_pack', 'hvx_tl2_unpack'):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            fn.restype = ctypes.c_int

    def encode(self, q):
        n, k = q.shape
        source = np.ascontiguousarray(q.T, dtype=np.int8)
        encoded = np.empty(((k + 23)//24) * ((n + 127)//128) * 640, np.uint8)
        if self.lib.hvx_tl2_pack(source.ctypes.data, k, n, encoded.ctypes.data):
            raise ValueError('Native TL2 pack failed')
        return encoded

    def verify(self, raw, q):
        n, k = q.shape
        encoded = np.frombuffer(raw, np.uint8)
        restored = np.empty((k, n), np.int8)
        if self.lib.hvx_tl2_unpack(encoded.ctypes.data, k, n, restored.ctypes.data):
            raise ValueError('Serialized TL2 decode failed')
        if not np.array_equal(restored.T, q):
            raise ValueError('Serialized TL2 differs from source trits')


def write_expert(dst, item, checkpoint, tl2):
    w = np.concatenate([checkpoint.read(name) for name in item['source']], axis=0)
    scale = np.abs(w).max(axis=1)
    q = np.sign(w).astype(np.int8)
    if not np.array_equal(w, q * scale[:, None]):
        raise ValueError(f'Non-ternary source expert: {item["name"]}')
    encoded = tl2.encode(q)
    sums = q.sum(axis=1, dtype=np.int32).astype('<f4')
    raw = encoded.tobytes() + scale.astype('<f4').tobytes() + sums.tobytes()
    if len(raw) != item['bytes']: raise ValueError('TL2 size mismatch')
    dst.seek(item['offset']); dst.write(raw); dst.flush()
    dst.seek(item['offset']); serialized = dst.read(item['bytes'])
    tl2.verify(serialized[:encoded.size], q)
    tail = np.frombuffer(serialized[encoded.size:], '<f4').reshape(2, -1)
    if not np.array_equal(tail[0], scale) or not np.array_equal(tail[1], sums):
        raise ValueError('Serialized expert scale/sum mismatch')
    return dict(name=item['name'], max_weight_error=0)


def write_fc(dst, item, checkpoint):
    k, n = item['shape']; maximum = energy = squared = 0.
    for c0 in range(0, n, 256):
        w = checkpoint.read(item['source'][0], c0, min(n, c0 + 256))
        q, scale, offsets = recover(w)
        tiles = pack(q)
        for kt, row in enumerate(tiles):
            dst.seek(item['offset'] + (kt * (n // 32) + c0 // 32) * 512)
            dst.write(row.tobytes())
        for at, v in [(k*n//2 + 4*c0, scale),
                      (k*n//2 + 4*(n+c0), q.sum(axis=1, dtype=np.int32).astype('<f4')),
                      (k*n//2 + 4*(2*n+c0), offsets)]:
            dst.seek(item['offset'] + at); dst.write(v.tobytes())
        dst.flush()
        verify_chunk(dst, item['offset'], k, n, c0, q, scale, offsets)
        error = (q.astype(np.int32) + offsets[:, None]) * scale[:, None] - w
        maximum = max(maximum, float(np.abs(error).max()))
        squared += np.square(error.astype(np.float64)).sum()
        energy += np.square(w.astype(np.float64)).sum()
    return dict(name=item['name'], max_bf16_roundtrip_error=0,
                max_fp32_grid_vs_bf16_error=maximum,
                fp32_grid_vs_bf16_rmse=float(np.sqrt(squared/(k*n))),
                fp32_grid_vs_bf16_relative_rmse=float(np.sqrt(squared/energy)) if energy else 0.)


def write_float(dst, item, checkpoint):
    name = item['source'][0]
    # Bound memory for the large embedding table.
    shape = checkpoint.header[name]['shape']
    count = shape[0]
    dst.seek(item['offset'])
    digest = hashlib.sha256()
    for first in range(0, count, 256):
        w = checkpoint.read(name, first, min(count, first + 256))
        if item['transform'] == 'transpose':
            # Routers are small and must be transposed as a whole.
            w = checkpoint.read(name).T
        elif item['transform'] == 'conv_reverse':
            w = checkpoint.read(name)[:, 0, ::-1].T
        raw = np.ascontiguousarray(w, dtype='<f4').tobytes()
        digest.update(raw); dst.write(raw)
        if item['transform'] != 'identity': break
    end = dst.tell()
    if end != item['offset'] + item['bytes']: raise ValueError('FP32 size mismatch')
    dst.flush(); dst.seek(item['offset'])
    remaining = item['bytes']; check = hashlib.sha256()
    while remaining:
        raw = dst.read(min(remaining, 4*1024**2))
        if not raw: raise EOFError('Truncated FP32 record')
        check.update(raw); remaining -= len(raw)
    if check.digest() != digest.digest(): raise ValueError('FP32 read-back mismatch')


def convert(source, output, tokenizer_source, library=None, cache_experts=16):
    source, output = source.resolve(), output.resolve()
    if output.exists(): raise FileExistsError(output)
    cfg = json.loads((source/'config.json').read_text())
    checkpoint = Checkpoint(source)
    try:
        plan, size = weight_plan(cfg, checkpoint)
        if not (tokenizer_source/'tokenizer.json').is_file():
            raise FileNotFoundError(tokenizer_source/'tokenizer.json')
        if not 1 <= cache_experts <= cfg['num_experts']:
            raise ValueError('Invalid resident expert count')
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < size + 1024**3:
            raise OSError('Insufficient free disk space')
        with tempfile.TemporaryDirectory(prefix='.lfm2-htp-', dir=output.parent) as work:
            work = Path(work)
            if library is None:
                library = work/'tl2_pack.so'
                root = Path(__file__).resolve().parents[1]
                subprocess.run(['cc', '-O3', '-shared', '-fPIC', '-I',
                    str(root/'nntrainer/tensor/htp_backend/hvx'),
                    str(root/'nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c'),
                    '-o', str(library)], check=True)
            tl2 = Tl2(library.resolve())
            binary = work/'nntr_lfm2_tl2_affine.bin'
            metrics = []; counts = Counter()
            with binary.open('xb+') as dst:
                dst.truncate(size)
                for i, item in enumerate(plan):
                    if item['dtype'] == 'TL2CX_WH':
                        metrics.append(write_expert(dst, item, checkpoint, tl2))
                    elif item['dtype'] == 'QS4CX_WH_AFFINE':
                        metrics.append(write_fc(dst, item, checkpoint))
                    else: write_float(dst, item, checkpoint)
                    counts[item['dtype']] += 1
                    if i % 128 == 0 or i == len(plan)-1:
                        print(f'{i+1}/{len(plan)} {item["name"]} verified', flush=True)
            now = checkpoint.path.stat()
            if (now.st_size, now.st_mtime_ns, now.st_ino) != (checkpoint.stamp.st_size, checkpoint.stamp.st_mtime_ns, checkpoint.stamp.st_ino):
                raise ValueError('Source checkpoint changed during conversion')
            config = dict(model_type='CausalLM', model_tensor_type='FP32-FP32',
                model_file_name=binary.name, lfm2_weight_format='htp_fused_v1',
                lfm2_execution_profile='htp_fused', embedding_dtype='FP32',
                fc_layer_dtype='QS4CX_WH_AFFINE', lmhead_dtype='QS4CX_WH_AFFINE',
                moe_layer_dtype='TL2CX_WH', short_conv_dtype='FP32',
                moe_engine='htp', attn_proj_engine='htp', dense_ffn_engine='htp',
                attention_engine='htp', attention_kv_dtype='q8_row', lmhead_engine='htp',
                batch_size=1, init_seq_len=512, max_seq_len=2048,
                num_to_generate=32, tokenizer_file='tokenizer.json', bad_word_ids=[],
                moe_cache_experts=cache_experts)
            report = dict(source=str(checkpoint.path),
                source_header_sha256=checkpoint.header_hash, source_bytes=now.st_size,
                model_file=binary.name, bytes=size, sha256=sha256(binary),
                storage_format='TL2CX_WH/1 + QS4CX_WH_AFFINE/1 + FP32',
                source_tensors_consumed=len(checkpoint.header), tensor_counts=dict(counts),
                expert_tensors_verified=counts['TL2CX_WH'],
                affine_fc_tensors_verified=counts['QS4CX_WH_AFFINE'],
                max_expert_weight_error=0, max_fc_bf16_roundtrip_error=0,
                requantization=False, short_conv_dtype='FP32',
                conv_layout='[3, hidden], newest tap first; source taps reversed',
                embedding='source values preserved as FP32 without sqrt(hidden) scaling',
                router='source BF16 promoted to FP32, transposed to [hidden, experts]; sigmoid+bias',
                activation='SwiGLU', device_validation='not performed',
                runtime_requirements=['Rebuilt ARM HTP runtime and matching v81 DSP skel',
                    f'Sigmoid router supporting {cfg["num_experts"]} experts',
                    'WH dense FFN intermediate slicing'],
                verification='every serialized TL2 trit/tail, WH nibble/tail and FP32 byte verified',
                tensors=plan, metrics=metrics)
            for name in ('config.json', 'generation_config.json'):
                shutil.copyfile(source/name, work/name)
            for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                if (tokenizer_source/name).is_file(): shutil.copyfile(tokenizer_source/name, work/name)
            (work/'nntr_config.json').write_text(json.dumps(config, indent=2)+'\n')
            (work/'conversion_report.json').write_text(json.dumps(report, indent=2)+'\n')
            # The host helper is not a runtime dependency or model artifact.
            if library.parent == work: library.unlink()
            work.rename(output)
            print(f'Published {output}: {size:,} bytes, SHA256 {report["sha256"]}', flush=True)
            return report
    finally: checkpoint.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokenizer-source', type=Path, help='Local matching base model tokenizer directory')
    ap.add_argument('--tl2-library', type=Path)
    ap.add_argument('--cache-experts', type=int, default=16)
    args = ap.parse_args()
    convert(args.source, args.output, args.tokenizer_source or args.source,
            args.tl2_library, args.cache_experts)
