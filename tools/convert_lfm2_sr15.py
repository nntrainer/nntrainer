#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert rotated LFM2 to a versioned weight pack and activation sidecar.

Weights use the existing native TL2/affine WH layouts. Q4_0 embedding is
GGML's reference 32-value/18-byte layout. DW3_INT4 is [C,2] packed unsigned
nibbles (newest tap first, unused fourth nibble zero), C float32 scales,
then C int32 offsets: w = (code + offset) * scale. A BF16 round restores
every source short-conv weight. The JSON manifest gives absolute offsets.

The rotation records are signed bytes: output signs then the row-major
small Hadamard basis. R = kron(basis, Sylvester(p)) * signs / sqrt(n).
Their magnitude is saved explicitly to reproduce the supplied F32 matrix.

Activation sidecar: <8sII header (NNTRACT1, version=1, count), followed by
<IIfiBBBB records (UTF-8 name bytes, mode mask, scale, original offset,
bitwidth, symmetric, INT dtype=1, reserved=0), then name bytes. Mode bits
are generation=1 and prefill=2. This stores supplied values, not inferred
runtime mappings. Full mapping coverage must be reported by a consumer.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np

from convert_gemma4_fc_wh import bf16_round, sha256
from convert_lfm2_htp import (Checkpoint, Tl2, weight_plan, write_expert,
                              write_fc, write_float)


def sylvester(n):
    h = np.ones((1, 1), np.int8)
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    if len(h) != n:
        raise ValueError('Sylvester size must be a power of two')
    return h


def rotation(source, name, n):
    path = source / (name + '_f32.npy')
    a = np.load(path, allow_pickle=False)
    if a.shape != (n, n) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Invalid rotation ' + name)
    mag = np.unique(np.abs(a))
    if len(mag) != 1 or not np.isclose(mag[0], 1 / math.sqrt(n), rtol=1e-6):
        raise ValueError('Not a normalized sign rotation: ' + name)
    s = np.sign(a).astype(np.int8)
    signs = s[0].copy()
    normalized = s * signs[None, :]
    for power in range(int(math.log2(n)), -1, -1):
        p = 1 << power
        if n % p or n // p > 44:
            continue
        basis = normalized[::p, ::p].copy()
        if np.array_equal(normalized, np.kron(basis, sylvester(p))):
            if not np.array_equal(basis.astype(np.int32) @ basis.T,
                                  np.eye(len(basis), dtype=np.int32) * len(basis)):
                raise ValueError('Nonorthogonal Hadamard basis')
            restored = (np.kron(basis, sylvester(p)) * signs).astype(np.float32) * mag[0]
            if not np.array_equal(restored, a):
                raise ValueError('Rotation does not round-trip')
            return dict(name=name, dtype='HADAMARD_SIGNED_I8', shape=[n, n],
                        sylvester_width=p, basis_width=len(basis),
                        magnitude=float(mag[0]), source_sha256=sha256(path),
                        transform='x @ R; output signs after Hadamard',
                        bytes=n+basis.size), signs.tobytes()+basis.tobytes()
    raise ValueError('Unsupported rotation factorization: ' + name)


def recover_small(w):
    """Find an affine grid even when a three-tap row never observes code 1."""
    mag = np.abs(w)
    peak = mag.max(axis=1)
    codes_out = np.zeros(w.shape, np.int32)
    scales = np.ones(len(w), np.float32)
    valid = peak == 0
    bits = mag.view(np.uint32) >> 16
    prev = ((np.maximum(bits, 1)-1) << 16).astype(np.uint32).view(np.float32)
    nex = ((bits+1) << 16).astype(np.uint32).view(np.float32)
    for top in range(1, 16):
        trial = np.where(peak > 0, peak/top, 1)
        codes = np.rint(w/trial[:, None]).astype(np.int32)
        den = np.maximum(np.abs(codes), 1)
        low = np.where(codes != 0, (prev.astype(np.float64)+mag)/(2*den), -np.inf).max(1)
        high = np.where(codes != 0, (nex.astype(np.float64)+mag)/(2*den), np.inf).min(1)
        low[peak == 0] = high[peak == 0] = 1
        scale = ((low+high)/2).astype(np.float32)
        fits = ((codes.max(1)-codes.min(1) <= 15) & (low <= high) &
                np.all(bf16_round(codes*scale[:, None]) == w, axis=1) & ~valid)
        codes_out[fits] = codes[fits]
        scales[fits] = scale[fits]
        valid |= fits
    if not valid.all():
        raise ValueError(f'{np.count_nonzero(~valid)} short-conv channels have no INT4 grid')
    offset = codes_out.min(1)
    return (codes_out-offset[:, None]).astype(np.uint8), scales, offset


def write_conv(dst, item, checkpoint):
    w = checkpoint.read(item['source'][0])[:, 0, ::-1].copy()
    q, s, o = recover_small(w)
    packed = np.stack([q[:, 0] | (q[:, 1] << 4), q[:, 2]], axis=1)
    raw = packed.tobytes()+s.astype('<f4').tobytes()+o.astype('<i4').tobytes()
    dst.seek(item['offset']); dst.write(raw); dst.flush()
    dst.seek(item['offset']); got = dst.read(item['bytes'])
    n = len(w); b = np.frombuffer(got[:2*n], np.uint8).reshape(n, 2)
    qs = np.stack([b[:, 0] & 15, b[:, 0] >> 4, b[:, 1] & 15], axis=1)
    scale = np.frombuffer(got[2*n:6*n], '<f4')
    offset = np.frombuffer(got[6*n:], '<i4')
    restored = bf16_round((qs.astype(np.int32)+offset[:, None])*scale[:, None])
    if got != raw or not np.array_equal(restored, w):
        raise ValueError('Short-conv readback mismatch')
    return dict(name=item['name'], max_bf16_roundtrip_error=0)


def q40(w):
    blocks = w.reshape(-1, 32)
    vmax = blocks[np.arange(len(blocks)), np.abs(blocks).argmax(1)]
    scale = vmax / np.float32(-8)
    inv = np.zeros_like(scale)
    np.divide(np.float32(1), scale, out=inv, where=scale != 0)
    q = np.minimum(15, (blocks*inv[:, None]+np.float32(8.5)).astype(np.int32)).astype(np.uint8)
    out = np.empty((len(blocks), 18), np.uint8)
    out[:, :2] = scale.astype('<f2').view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out


def decode_q40(raw):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 18)
    s = b[:, :2].copy().reshape(-1).view('<f2').astype(np.float32)
    q = np.concatenate([b[:, 2:] & 15, b[:, 2:] >> 4], axis=1).astype(np.int32)-8
    return q*s[:, None]


def write_embedding(dst, item, checkpoint):
    n, k = item['shape']; maximum = squared = energy = 0.
    dst.seek(item['offset'])
    for first in range(0, n, 256):
        w = checkpoint.read(item['source'][0], first, first+256)
        raw = q40(w).tobytes(); at = dst.tell()
        dst.write(raw); dst.flush(); dst.seek(at)
        got = dst.read(len(raw))
        if got != raw:
            raise ValueError('Q4_0 readback mismatch')
        error = decode_q40(got).reshape(w.shape)-w
        maximum = max(maximum, float(np.abs(error).max()))
        squared += np.square(error.astype(np.float64)).sum()
        energy += np.square(w.astype(np.float64)).sum()
    return dict(name=item['name'], max_abs_error=maximum,
                rmse=float(np.sqrt(squared/(n*k))),
                relative_rmse=float(np.sqrt(squared/energy)))


def encode_activations(directory, target):
    entries = {}; sources = {}
    for mode, bit in [('generation', 1), ('prefill', 2)]:
        path = directory / f'lfm2_24b_w4a8kv8_{mode}.encodings'
        data = json.loads(path.read_text()); sources[mode] = sha256(path)
        seen = set()
        for e in data['activation_encodings']:
            name = e['name']
            if name in seen:
                raise ValueError('Duplicate activation name')
            seen.add(name)
            if (e['enc_type'] != 'PER_TENSOR' or e['dtype'] != 'INT' or
                    e['bw'] not in (8, 16) or len(e['scale']) != 1 or
                    len(e['offset']) != 1):
                raise ValueError('Unsupported activation encoding: ' + name)
            s = np.float32(e['scale'][0]); o = int(e['offset'][0])
            if not np.isfinite(s) or s <= 0 or not -(1 << e['bw']) < o <= 0:
                raise ValueError('Invalid activation scale/offset')
            key = (name, float(s), o, e['bw'], bool(e['is_sym']))
            entries[key] = entries.get(key, 0) | bit
    raw = bytearray(struct.pack('<8sII', b'NNTRACT1', 1, len(entries)))
    for (name, s, o, bw, sym), mask in sorted(entries.items()):
        name = name.encode('utf-8')
        raw += struct.pack('<IIfiBBBB', len(name), mask, s, o, bw, sym, 1, 0)+name
    target.write_bytes(raw)
    decoded = read_activations(target)
    if decoded != entries:
        raise ValueError('Activation binary round-trip mismatch')
    return dict(records=len(entries), source_sha256=sources,
                bitwidth_counts=dict(Counter(k[3] for k in entries)),
                bytes=len(raw), sha256=sha256(target),
                mapping='Original tensor names preserved; runtime mapping required')


def read_activations(path):
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError('Truncated activation header')
    magic, version, count = struct.unpack_from('<8sII', data)
    if magic != b'NNTRACT1' or version != 1:
        raise ValueError('Unsupported activation format')
    at = 16; result = {}
    for _ in range(count):
        if at+20 > len(data):
            raise ValueError('Truncated activation record')
        n, mask, s, o, bw, sym, dtype, reserved = struct.unpack_from('<IIfiBBBB', data, at)
        at += 20
        if (n == 0 or at+n > len(data) or mask not in (1, 2, 3) or
                not math.isfinite(s) or s <= 0 or bw not in (8, 16) or
                sym not in (0, 1) or dtype != 1 or reserved):
            raise ValueError('Invalid activation record')
        name = data[at:at+n].decode('utf-8'); at += n
        key = (name, s, o, bw, bool(sym))
        if key in result:
            raise ValueError('Duplicate activation record')
        result[key] = mask
    if at != len(data):
        raise ValueError('Trailing activation bytes')
    return result


def convert(source, encodings, output, tokenizer, rotation_mode="both"):
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((source/'config.json').read_text())
    checkpoint = Checkpoint(source)
    try:
        plan, _ = weight_plan(cfg, checkpoint)
        cursor = 0
        for item in plan:
            if item['name'] == 'embedding0:Embedding':
                item.update(dtype='Q4_0', bytes=math.prod(item['shape'])//32*18)
            elif item['name'].endswith('_conv:conv'):
                item.update(dtype='DW3_INT4', bytes=cfg['hidden_size']*10)
            elif item['name'].endswith('_sparse_moe:gate'):
                k, n = item['shape']
                item.update(dtype='QS4CX_WH_AFFINE', bytes=k*n//2+12*n,
                            transform='identity')
            item['offset'] = cursor; cursor += item['bytes']
        rotations = []
        for name, n in [('R_in', cfg['hidden_size']), ('R_down', cfg['moe_intermediate_size'])]:
            info, raw = rotation(source, name, n)
            info['offset'] = cursor; cursor += len(raw)
            rotations.append((info, raw))
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < cursor+1024**3:
            raise OSError('Insufficient disk space')
        with tempfile.TemporaryDirectory(prefix='.sr15-', dir=output.parent) as tmp:
            work = Path(tmp); library = work/'tl2.so'
            root = Path(__file__).resolve().parents[1]
            subprocess.run(['cc', '-O3', '-shared', '-fPIC', '-I',
                str(root/'nntrainer/tensor/htp_backend/hvx'),
                str(root/'nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c'),
                '-o', str(library)], check=True)
            tl2 = Tl2(library); metrics = []
            binary = work/'LFM2-24B-A2B-Ternary-INT4.bin'
            writers = {'Q4_0': write_embedding, 'DW3_INT4': write_conv,
                       'QS4CX_WH_AFFINE': write_fc}
            with binary.open('xb+') as dst:
                for i, item in enumerate(plan):
                    if item['dtype'] == 'TL2CX_WH':
                        metric = write_expert(dst, item, checkpoint, tl2)
                    elif item['dtype'] in writers:
                        metric = writers[item['dtype']](dst, item, checkpoint)
                    else:
                        write_float(dst, item, checkpoint); metric = None
                    if metric: metrics.append(metric)
                    if i % 128 == 0:
                        print(f'{i+1}/{len(plan)} {item["name"]} verified', flush=True)
                for info, raw in rotations:
                    dst.seek(info['offset']); dst.write(raw); dst.flush()
                    dst.seek(info['offset'])
                    if dst.read(len(raw)) != raw: raise ValueError('Rotation readback')
            if binary.stat().st_size != cursor:
                raise ValueError('Weight binary length mismatch')
            stamp = checkpoint.path.stat()
            if (stamp.st_size, stamp.st_mtime_ns) != (checkpoint.stamp.st_size, checkpoint.stamp.st_mtime_ns):
                raise ValueError('Source changed during conversion')
            activation = encode_activations(encodings, work/'activation_scales.bin')
            activation['runtime_usage'] = 'unused: dynamic activation quantization'
            report = dict(format='nntr_lfm2_sr15_v2', model_name='LFM2-24B-A2B-Ternary-INT4', model_file=binary.name,
                bytes=cursor, sha256=sha256(binary), source=str(source.resolve()),
                source_header_sha256=checkpoint.header_hash,
                tensors=plan, rotations=[i for i, _ in rotations],
                tensor_counts=dict(Counter(i['dtype'] for i in plan)),
                activation=activation, metrics=metrics,
                runtime_validation='not performed')
            (work/'conversion_report.json').write_text(json.dumps(report, indent=2)+'\n')
            for name in ['config.json', 'generation_config.json', 'chat_template.jinja']:
                shutil.copyfile(source/name, work/name)
            for name in ['tokenizer.json', 'tokenizer_config.json']:
                shutil.copyfile(tokenizer/name, work/name)
            config = dict(model_type='CausalLM', model_tensor_type='FP32-FP32',
                model_file_name=binary.name, activation_quantization='dynamic',
                lfm2_weight_format='sr15_v2', lfm2_execution_profile='htp_fused',
                embedding_dtype='Q4_0', fc_layer_dtype='QS4CX_WH_AFFINE',
                lmhead_dtype='QS4CX_WH_AFFINE', router_dtype='QS4CX_WH_AFFINE',
                moe_layer_dtype='TL2CX_WH', short_conv_dtype='DW3_INT4',
                moe_engine='htp', attn_proj_engine='htp', dense_ffn_engine='htp',
                attention_engine='htp', attention_kv_dtype='q8_row', lmhead_engine='htp',
                moe_hadamard=rotation_mode, moe_hadamard_scope='all',
                batch_size=1, init_seq_len=256, max_seq_len=2048,
                num_to_generate=32, tokenizer_file='tokenizer.json', bad_word_ids=[],
                sample_input='<|im_start|>user\nWhat is the capital of France? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n',
                moe_cache_experts=16)
            (work/'nntr_config.json').write_text(json.dumps(config, indent=2)+'\n')
            library.unlink(); work.rename(output)
            print(f'Published {output}: {cursor:,} bytes; {report["sha256"]}', flush=True)
    finally:
        checkpoint.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--encodings', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokenizer-source', type=Path, required=True)
    ap.add_argument('--rotation', choices=['off', 'up', 'down', 'both'], default='both',
                    help='Online rotation mode; the rotated SR15 checkpoint requires both for correct inference')
    args = ap.parse_args()
    convert(args.source, args.encodings, args.output, args.tokenizer_source, args.rotation)
