#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file quantize_fp32_to_q8_0.py
# @brief Produce the W8A16 backbone weights (Q8_0) from a corrected FP32
#        safetensors, mirroring nntrainer's ON-DISK byte layout EXACTLY.
#
#        nntrainer stores a Q8_0 conv filter as a stream of block_q8_0x4
#        super-blocks (NOT vanilla 34-byte block_q8_0 rows), produced by
#        repack_q8_0 (nntrainer/tensor/cpu_backend/conv_indirect.h:381):
#
#          block_q8_0x4 { uint16_t d[4]; int8_t qs[128]; }  // 136 bytes
#
#        gathering 4 output-channel rows per super-block. The loader
#        (TensorBase::read) does a RAW byte copy with no de-interleave, and
#        the runtime GEMM (Q8_0_Tensor dot/convQ4_0Indirect) consumes the
#        bytes as block_q8_0x4 directly. Writing vanilla 34-byte blocks
#        therefore makes the kernel read int8-quant bytes as fp16 scales,
#        yielding fp16 NaN scales -> NaN propagation on device. (This was the
#        root cause of the W8A16 NaN; the quantization MATH was always
#        correct.)
#
#        Pipeline (mirror of Conv2DLayer::save, conv2d_layer.cpp:2199-2211):
#          1. quantize each output row into vanilla block_q8_0 (d=amax/127,
#             qs=round(x/d), row-major over CRS=in_ch*kh*kw) -- this is
#             quantize_row_q8_0_ref (nntr_ggml_impl_quant.cpp:497).
#          2. repack_q8_0 the vanilla stream into block_q8_0x4.
#
#        Eligibility MUST match the set the graph's convQuantEligible /
#        save-time guard quantizes (groups==1, out_ch%32==0, CRS%32==0), AND
#        the exact key set the deployed graph expects. To guarantee a
#        load-compatible file, the quantized key set is taken from a
#        reference working-format q8_0 file (--ref-q8, default ref2) when
#        available, falling back to the predicate. Depthwise/grouped convs
#        and block-misaligned filters stay FP32.
#
#        The safetensors header is written by hand (not safetensors.save_file)
#        so it carries the fields the C++ saver emits and the loader checks:
#          __metadata__ = {"format":"nntrainer","nntr_format":"nntr-safetensors-v1"}
#          Q8_0 entry: dtype="U8", shape=[N*K/32*34], nntr_dtype="Q8_0",
#                      nntr_shape=[1,1,K,N]   (K=CRS, N=out_ch)
#          F32 entry:  dtype="F32", shape=<original 4D/1D shape>  (no nntr fields)
#
# Usage:
#   python quantize_fp32_to_q8_0.py \
#       --fp32 /path/fastvit_keyword.safetensors \
#       --out  /path/fastvit_keyword_q8_0.safetensors \
#       --ref-q8 /path/existing_fastvit_keyword_q8_0.safetensors

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


def quantize_row_q8_0(row):
    """Mirror quantize_row_q8_0_ref -> vanilla block_q8_0 stream (bytes) for one row."""
    assert row.size % QK == 0, row.size
    nb = row.size // QK
    out = bytearray()
    for b in range(nb):
        block = row[b * QK:(b + 1) * QK].astype(np.float32)
        amax = float(np.abs(block).max())
        d = amax / 127.0 if amax > 0.0 else 0.0
        id_ = 1.0 / d if d != 0.0 else 0.0
        # round-to-nearest, clip to [-127,127] (avoid -128 wrap that some kernels treat as NaN-source)
        qs = np.clip(np.rint(block * id_), -127, 127).astype(np.int8)
        out += struct.pack('<H', fp32_to_fp16_u16(d))
        out += qs.tobytes()
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


def quantize_filter_x4(w):
    """w: FP32 [out_ch, in_ch, kh, kw]. Returns (x4_bytes, N, K)."""
    N = w.shape[0]
    K = int(w.size // N)
    assert K == w.shape[1] * w.shape[2] * w.shape[3]
    flat = w.reshape(N, K).astype(np.float32)
    plain = bytearray()
    for r in range(N):
        plain += quantize_row_q8_0(flat[r])
    return repack_q8_0(bytes(plain), N, K), N, K


def conv_quant_eligible(shape, groups=1):
    out_ch = shape[0]
    crs = int(shape[1]) * int(shape[2]) * int(shape[3])
    return groups == 1 and out_ch % 32 == 0 and crs % 32 == 0


def read_ref_u8_keys(ref_path):
    """Return {key: nntr_shape} for U8/Q8_0 entries in a working-format ref file."""
    if ref_path is None:
        return None
    import os
    if not os.path.exists(ref_path):
        return None
    with open(ref_path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        header = json.loads(fh.read(n))
    u8 = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        if v.get('dtype') == 'U8' and v.get('nntr_dtype') == 'Q8_0':
            u8[k] = v['nntr_shape']  # [1,1,K,N]
    return u8


def write_safetensors(path, entries, metadata):
    """Hand-write a safetensors file with nntr fields + metadata.

    entries: list of dicts {name, dtype('U8'|'F32'), shape(list), data(bytes),
                            nntr_dtype(optional), nntr_shape(optional)}
    """
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
    # safetensors padding: header must be padded so total header (8+len) keeps
    # data aligned; the C++ loader does not require padding, but the reference
    # files are unpadded. Match reference: no padding.
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(blob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp32', required=True, help='corrected FP32 safetensors')
    ap.add_argument('--out', required=True, help='output Q8_0 safetensors')
    ap.add_argument('--ref-q8', default='/home/seungbaek/projects/nntrainer_ref2/'
                        'Applications/causallm/models/FastViTKeyword/res/'
                        'fastvit_keyword_q8_0.safetensors',
                    help='working-format q8_0 file to read the quantized key set from')
    args = ap.parse_args()

    ref_u8 = read_ref_u8_keys(args.ref_q8)

    entries = []
    n_q8 = n_fp32_filter = n_other = 0
    with safe_open(args.fp32, framework='numpy') as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith(':filter') and t.ndim == 4:
                eligible = conv_quant_eligible(t.shape)
                use_q8 = (k in ref_u8) if ref_u8 is not None else eligible
                if use_q8:
                    x4, N, K = quantize_filter_x4(t.astype(np.float32))
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
          f"({n_q8} Q8_0/x4 filters, {n_fp32_filter} FP32 filters, {n_other} other)")
    if ref_u8 is not None:
        print(f"quantized key set taken from ref: {args.ref_q8} ({len(ref_u8)} U8 keys)")
    else:
        print("WARNING: ref-q8 not found; used eligibility predicate only "
              "(may not match graph's exact quantized set)")


if __name__ == '__main__':
    main()
