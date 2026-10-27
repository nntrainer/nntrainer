#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file quantize_fp32_to_qint8.py
# @brief Produce per-channel Q8_0 (QINT8) weights from an FP32 safetensors.
#
#        Unlike regular Q8_0 (per-32-element-block scale), this produces
#        PER-CHANNEL Q8_0: every block within an output-channel row carries
#        the SAME fp16 scale d (amax/127 over the entire row). This is the
#        "pch" file format that __ggml_q8ch_prepare_conv_weight detects at
#        load time and takes the fast path (int8 straight-through, no FP32
#        round-trip), matching the per-channel int8 scheme used by the W8A8
#        per-channel kernel (NNTR_W8A8 + NNTR_W8A8_PERCH).
#
#        The on-disk layout is identical to regular Q8_0 (block_q8_0x4
#        super-blocks), so the loader and graph see "Q8_0" — the difference
#        is purely in the scale uniformity, which the runtime detects.
#
#        This mirrors __ggml_quantize_q8_0_per_channel in
#        ggml_interface.cpp:652, which does the same per-channel quantization
#        at C++ level (used for on-the-fly FP32 weight conversion).
#
#        Eligibility matches the graph's convQuantEligible:
#          groups==1, out_ch%32==0, CRS%32==0
#        Non-eligible filters stay FP32.
#
# Usage:
#   python quantize_fp32_to_qint8.py \
#       --fp32 /path/yolov7_tiny.safetensors \
#       --out  /path/yolov7_tiny_qint8.safetensors

import argparse
import json
import struct

import numpy as np
from safetensors import safe_open

QK = 32
BLOCK_Q8_0 = 34          # 2 (fp16 d) + 32 (int8 qs)
SUPERBLOCK_Q8_0X4 = 136  # 8 (4x fp16 d) + 128 (int8 qs)


def fp32_to_fp16_u16(x):
    return int(np.float16(x).view(np.uint16))


def quantize_row_q8_0_per_channel(row):
    """Per-channel Q8_0 quantization for one output-channel row.

    Mirrors __ggml_quantize_q8_0_per_channel (ggml_interface.cpp:652):
    - amax over the ENTIRE row (not per 32-element block)
    - d = amax / 127, stored as fp16
    - All blocks in this row get the SAME d
    - qs = round(x / d_used), clipped to [-127, 127]
    """
    assert row.size % QK == 0, row.size
    nb = row.size // QK
    out = bytearray()

    # Per-channel: amax over the entire row
    amax = float(np.abs(row).max())
    d = amax / 127.0 if amax > 0.0 else 1.0
    d16 = fp32_to_fp16_u16(d)
    d_used = np.float16(d).astype(np.float32)  # exact runtime value after fp16 round-trip
    inv = 1.0 / d_used if d_used > 0.0 else 0.0

    # Quantize all elements
    qs = np.clip(np.rint(row * inv), -127, 127).astype(np.int8)

    # Write blocks — all with the same d
    for j in range(nb):
        out += struct.pack('<H', d16)
        out += qs[j * QK:(j + 1) * QK].tobytes()
    return bytes(out)


def repack_q8_0(plain, N, K):
    """Exact mirror of conv_indirect.h:381 repack_q8_0.

    plain: vanilla block_q8_0 stream of N rows x (K/32) blocks (N*nb*34 bytes).
    Returns block_q8_0x4 stream (same total byte count, different order).
    """
    assert N % 4 == 0 and K % QK == 0, f"N={N} K={K} (need N%4==0, K%32==0)"
    nb = K // QK
    out = bytearray()
    for sc in range(N // 4):
        for j in range(nb):
            d_vals = []
            qs_rows = [None] * 4
            for r in range(4):
                p_off = ((sc * 4 + r) * nb + j) * BLOCK_Q8_0
                d_vals.append(plain[p_off:p_off + 2])
                qs_rows[r] = plain[p_off + 2:p_off + BLOCK_Q8_0]
            out += b''.join(d_vals)                       # 4 fp16 scales (8 B)
            sb = np.zeros(128, dtype=np.int8)             # 128 int8 quants
            for r in range(4):
                q = np.frombuffer(qs_rows[r], dtype=np.int8)
                for sub in range(4):
                    sb[sub * 32 + r * 8:sub * 32 + r * 8 + 8] = q[sub * 8:sub * 8 + 8]
            out += sb.tobytes()
    return bytes(out)


def quantize_filter_per_channel_x4(w):
    """w: FP32 [out_ch, in_ch, kh, kw]. Returns (x4_bytes, N, K).

    Per-channel quantization: one scale per output channel row.
    """
    N = w.shape[0]
    K = int(w.size // N)
    assert K == w.shape[1] * w.shape[2] * w.shape[3]
    flat = w.reshape(N, K).astype(np.float32)
    plain = bytearray()
    for r in range(N):
        plain += quantize_row_q8_0_per_channel(flat[r])
    return repack_q8_0(bytes(plain), N, K), N, K


def conv_quant_eligible(shape, groups=1):
    out_ch = shape[0]
    crs = int(shape[1]) * int(shape[2]) * int(shape[3])
    return groups == 1 and out_ch % 32 == 0 and crs % 32 == 0


def write_safetensors(path, entries, metadata):
    """Hand-write a safetensors file with nntr fields + metadata."""
    header = {'__metadata__': metadata}
    offset = 0
    blob = bytearray()
    for e in entries:
        nbytes = len(e['data'])
        header[e['name']] = {
            'dtype': e['dtype'],
            'shape': list(e['shape']),
            'data_offsets': [offset, offset + nbytes],
        }
        if e.get('nntr_dtype') is not None:
            header[e['name']]['nntr_dtype'] = e['nntr_dtype']
        if e.get('nntr_shape') is not None:
            header[e['name']]['nntr_shape'] = list(e['nntr_shape'])
        blob += e['data']
        offset += nbytes
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(blob))


def main():
    ap = argparse.ArgumentParser(
        description='Convert FP32 safetensors to per-channel Q8_0 (QINT8) safetensors')
    ap.add_argument('--fp32', required=True, help='FP32 safetensors input')
    ap.add_argument('--out', required=True, help='Output QINT8 safetensors path')
    args = ap.parse_args()

    entries = []
    n_q8 = n_fp32_filter = n_other = 0
    with safe_open(args.fp32, framework='numpy') as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith(':filter') and t.ndim == 4:
                eligible = conv_quant_eligible(t.shape)
                if eligible:
                    x4, N, K = quantize_filter_per_channel_x4(t.astype(np.float32))
                    entries.append({
                        'name': k, 'dtype': 'U8', 'shape': [len(x4)],
                        'data': x4, 'nntr_dtype': 'Q8_0',
                        'nntr_shape': [1, 1, K, N],
                    })
                    n_q8 += 1
                else:
                    w = np.ascontiguousarray(t.astype(np.float32))
                    entries.append({
                        'name': k, 'dtype': 'F32', 'shape': list(w.shape),
                        'data': w.tobytes(),
                    })
                    n_fp32_filter += 1
            elif k.endswith(':filter'):
                w = np.ascontiguousarray(t.astype(np.float32))
                entries.append({'name': k, 'dtype': 'F32', 'shape': list(w.shape),
                                'data': w.tobytes()})
                n_fp32_filter += 1
            elif k.endswith(':bias') or k.endswith(':gamma') or k.endswith(':beta'):
                w = np.ascontiguousarray(t.astype(np.float32))
                entries.append({'name': k, 'dtype': 'F32', 'shape': list(w.shape),
                                'data': w.tobytes()})
                n_other += 1
            else:
                w = np.ascontiguousarray(t)
                dt = 'F32' if w.dtype == np.float32 else 'U8'
                entries.append({'name': k, 'dtype': dt, 'shape': list(w.shape),
                                'data': w.tobytes()})
                n_other += 1

    metadata = {'format': 'nntrainer', 'nntr_format': 'nntr-safetensors-v1'}
    write_safetensors(args.out, entries, metadata)
    print(f"wrote {len(entries)} tensors to {args.out} "
          f"({n_q8} per-channel Q8_0 filters, {n_fp32_filter} FP32 filters, {n_other} other)")


if __name__ == '__main__':
    main()
