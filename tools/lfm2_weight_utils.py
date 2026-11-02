#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint reading and native record writers for the LFM2 SR15 converter."""
import ctypes
import hashlib
import math

import numpy as np
from htp_weight_utils import Checkpoint, write_affine_wh


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
            expected = [[h, 1, 3]] if dtype == 'DW3_INT4' else [shape]
            if dtype == 'Q4_0':
                size = math.prod(shape) // 32 * 18
            elif dtype == 'DW3_INT4':
                size = h * 10
            else:
                size = math.prod(shape) * 4
        for key, want in zip(names, expected):
            if checkpoint.header[key]['shape'] != want:
                raise ValueError(f'Shape mismatch: {key}: expected {want}')
        plan.append(dict(name=name, source=names, shape=shape, dtype=dtype,
                         transform=transform, offset=cursor, bytes=size))
        cursor += size

    add('embedding0:Embedding', 'model.embed_tokens.weight', [cfg['vocab_size'], h], 'Q4_0')
    for l, kind in enumerate(cfg['layer_types']):
        p, src = f'layer{l}', f'model.layers.{l}.'
        if kind == 'conv':
            add(p + '_conv_norm:gamma', src + 'operator_norm.weight', [h])
            add(p + '_conv:in_proj', src + 'conv.in_proj.weight', [h, 3*h], 'QS4CX_WH_AFFINE')
            add(p + '_conv:conv', src + 'conv.conv.weight', [3, h], 'DW3_INT4', transform='conv_reverse')
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
            add(m + 'gate', src + 'feed_forward.gate.weight', [h, cfg['num_experts']], 'QS4CX_WH_AFFINE')
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
    n = item['shape'][1]
    rows = ((c0, checkpoint.read(item['source'][0], c0, min(n, c0 + 256)))
            for c0 in range(0, n, 256))
    return write_affine_wh(dst, item, rows)


def write_float(dst, item, checkpoint):
    name = item['source'][0]
    shape = checkpoint.header[name]['shape']
    count = shape[0]
    dst.seek(item['offset'])
    digest = hashlib.sha256()
    for first in range(0, count, 256):
        w = checkpoint.read(name, first, min(count, first + 256))
        raw = np.ascontiguousarray(w, dtype='<f4').tobytes()
        digest.update(raw); dst.write(raw)
    end = dst.tell()
    if end != item['offset'] + item['bytes']: raise ValueError('FP32 size mismatch')
    dst.flush(); dst.seek(item['offset'])
    remaining = item['bytes']; check = hashlib.sha256()
    while remaining:
        raw = dst.read(min(remaining, 4*1024**2))
        if not raw: raise EOFError('Truncated FP32 record')
        check.update(raw); remaining -= len(raw)
    if check.digest() != digest.digest(): raise ValueError('FP32 read-back mismatch')
