#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   quantize_q4_0_conv.py
@brief  Offline ggml Q4_0 block quantization of conv weights for YOLOv11m.

Default mode: quantizes eligible 1x1 conv filters (kh==kw==1).
With --all   : also quantizes eligible 3x3 and other groups=1 conv filters
               (kh*kw > 1), using the same Q4_0 block scheme but with the
               flattened [CRS=in_ch*kh*kw, out_ch] layout.

Reads yolov11m_fused.safetensors (FP32), quantizes every eligible conv
filter to ggml Q4_0 + repacks into nntrainer's q4_0x4 (ARM) or q4_0x8 (x86)
interleaved layout, and writes yolov11m_fused_q40.safetensors (1x1 only) or
yolov11m_fused_q40_all.safetensors (all conv, with --all).

Quantization scheme (ggml Q4_0, matches nntr_ggml_impl_quant.cpp):
  - Block size QK4_0 = 32 elements per block.
  - Per block of 32 values: amax = max(|x|), d = amax / -8.
  - id = 1/d (or 0 when d==0).
  - qi = clip(round(x * id) + 8, 0, 15)  (unsigned 4-bit, range [0, 15])
  - Nibble packing (low nibble = first 16 elements, high nibble = last 16):
      qs[j] = qi[j] | (qi[j + 16] << 4)   for j in [0, 16)
  - d stored as fp16 (2 bytes, little-endian), then qs[16] = 18 bytes per block.

Layout match (1x1 conv, matches original behaviour):
  Runtime path (layer_devel.h, quantize_q4_0 / repack_q4_0):
    1. weight tensor dim for quantized 1x1 conv = [1, 1, K=in_ch, N=out_ch]
       (Conv2DLayer::finalize sets this when Q4_0 dtype and kh==kw==1)
    2. weight.transpose("0:2:1") -> [1, 1, N=out_ch, K=in_ch]
    3. quantize_q4_0(transposed, nrow=N, n_per_row=K)
       -> N rows of K/32 plain block_q4_0 blocks
    4. repack_q4_0(plain, N, K, target_isa) -> q4_0xINTERLEAVE in-place
    5. save -> [1, 1, K, N] bytes
  Load path (Q4_0_Tensor::read): reads getMemoryBytes() bytes directly,
  no repack on load. File must contain already-repacked data.

  This tool replicates steps 2-5 offline from the FP32 weight.
  Input filter shape (1x1): [out_ch, in_ch, 1, 1] in safetensors.
  Weight is squeezed to [out_ch, in_ch] (= the transpose step).
  Output header shape: [1, 1, K=in_ch, N=out_ch].

Layout match (general kh*kw > 1 conv, --all mode):
  Input filter shape: [out_ch, in_ch, kh, kw] in safetensors (C-contiguous).
  Flatten: w.reshape(out_ch, CRS) where CRS = in_ch * kh * kw.
    Flatten order: (in_ch outermost, then kh, then kw) = row-major over the
    last three axes. K-index = ic*kh*kw + ki*kw + kj.
    This matches nntrainer's im2col column ordering exactly:
      im2col loops: channel (outer) -> h -> w (inner), same K-index formula.
  The reshaped [out_ch, CRS] = [N, K] is quantized identically to 1x1.
  Output header shape: [1, 1, K=CRS, N=out_ch] (same convention as 1x1).
  nntr_dtype = "Q4_0", nntr_shape = [1, 1, CRS, out_ch].

Exclusion rules:
  - depthwise convs (groups > 1, identified by 'dw:' in name)
  - out_ch == 1 (degenerate)
  - out_ch % 32 != 0  (Q4_0 block alignment, N side)
  - CRS % 32 != 0     (Q4_0 block alignment, K side; for 1x1 CRS = in_ch)
  - out_ch % interleave != 0  (repack alignment)
  All excluded tensors are kept FP32.

Usage:
  # 1x1 only (default, backward-compatible):
  python quantize_q4_0_conv.py \\
      --input  ../res/yolov11m_fused.safetensors \\
      --output ../res/yolov11m_fused_q40.safetensors \\
      --target arm

  # All groups=1 conv (1x1 + 3x3 + stride-2 etc.):
  python quantize_q4_0_conv.py \\
      --input  ../res/yolov11m_fused.safetensors \\
      --output ../res/yolov11m_fused_q40_all.safetensors \\
      --target arm --all

  # Dry-run: show which tensors would be quantized without writing
  python quantize_q4_0_conv.py --input ../res/yolov11m_fused.safetensors \\
      --all --dry-run

References:
  nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_quant.cpp
  nntrainer/tensor/q4_0_utils.h   (block_q4_0, block_q4_0x4, block_q4_0x8)
  nntrainer/layers/layer_devel.h  (Q4_0 save path)
  nntrainer/layers/conv2d_layer.cpp (quant_matmul_filter layout, im2col column order)
  Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py (repack impl)
"""

import argparse
import json
import struct
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Q4_0 constants
# ---------------------------------------------------------------------------
QK4_0 = 32
Q4_0_BLOCK_BYTES = 18   # 2 bytes fp16 d + 16 bytes qs

# Safetensors dtype strings for Q4_0 (opaque blob stored as U8; nntr_dtype=Q4_0)
SAFETENSORS_DTYPE_Q4_0 = "U8"
NNTR_DTYPE_Q4_0 = "Q4_0"

# Q8_0 constants (8-bit weight variant — higher accuracy than Q4_0)
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34   # 2 bytes fp16 d + 32 int8 qs
SAFETENSORS_DTYPE_Q8_0 = "U8"
NNTR_DTYPE_Q8_0 = "Q8_0"


# ---------------------------------------------------------------------------
# Safetensors I/O helpers (reused from quantize_int4_conv.py)
# ---------------------------------------------------------------------------

def parse_safetensors(path: str):
    """Parse safetensors file. Returns (header_dict, raw_data_bytes, header_size)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        data = f.read()
    header = json.loads(header_json)
    return header, data, header_size


def write_safetensors(tensors: dict, path: str):
    """Write tensors to a safetensors file.

    tensors: dict[name -> {"dtype": str, "shape": list[int], "data": bytes,
                           "nntr_dtype": str (opt), "nntr_shape": list (opt)}]
    """
    blob = bytearray()
    header = {"__metadata__": {"format": "nntrainer", "nntr_format": "nntr-safetensors-v1"}}
    offset = 0
    for name, t in tensors.items():
        data = t["data"]
        end = offset + len(data)
        entry = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, end],
        }
        if "nntr_dtype" in t:
            entry["nntr_dtype"] = t["nntr_dtype"]
        if "nntr_shape" in t:
            entry["nntr_shape"] = t["nntr_shape"]
        header[name] = entry
        blob += data
        offset = end

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(blob)


# ---------------------------------------------------------------------------
# Q4_0 quantization (matches quantize_row_q4_0_ref in nntr_ggml_impl_quant.cpp)
# ---------------------------------------------------------------------------

def quantize_q4_0(w2d: np.ndarray) -> bytes:
    """Quantise float32 array [N, K] to raw Q4_0 block stream (plain, pre-repack).

    Each row of K elements produces K/32 blocks.
    Block layout: [fp16 d (2 bytes)] [qs[16] uint8] = 18 bytes.

    Matches quantize_row_q4_0_ref:
      d = max(x) / -8  (max = the value with max abs, preserving sign)
      qi = clip(round(x / d) + 8, 0, 15)
      qs[j] = qi[j] | (qi[j+16] << 4)   for j in [0, 16)
    """
    assert w2d.ndim == 2
    N, K = w2d.shape
    assert K % QK4_0 == 0, f"K={K} must be divisible by QK4_0={QK4_0}"

    w2d = w2d.astype(np.float32)
    nblocks_per_row = K // QK4_0
    total_blocks = N * nblocks_per_row

    # Reshape to (total_blocks, 32)
    blocks = w2d.reshape(total_blocks, QK4_0)

    # Find the maximum-magnitude value in each block (preserving sign, like ggml)
    abs_vals = np.abs(blocks)
    max_idx = np.argmax(abs_vals, axis=1)          # (nb,)
    max_val = blocks[np.arange(total_blocks), max_idx]  # signed max

    d = (max_val / -8.0).astype(np.float32)        # (nb,)
    id_ = np.where(d != 0.0, 1.0 / d, np.float32(0.0))

    # Quantize: round(x * id) + 8, clip to [0, 15]
    q = np.clip(np.rint(blocks * id_[:, None]) + 8.0, 0, 15).astype(np.uint8)

    # Nibble packing: qs[j] = q[j] | (q[j+16] << 4) for j in [0,16)
    low  = q[:, :16]   # (nb, 16)
    high = q[:, 16:]   # (nb, 16)
    qs = (low | (high << 4)).astype(np.uint8)      # (nb, 16)

    # Encode d as fp16 (2 bytes, little-endian)
    d_fp16 = d.astype(np.float16).view(np.uint16)
    d_bytes = d_fp16.view(np.uint8).reshape(total_blocks, 2)

    # Interleave: [2 bytes d][16 bytes qs] per block
    raw = np.concatenate([d_bytes, qs], axis=1).astype(np.uint8)  # (nb, 18)
    return raw.tobytes()


# ---------------------------------------------------------------------------
# Q8_0 quantization (matches quantize_row_q8_0_ref in ggml)
# ---------------------------------------------------------------------------

def quantize_q8_0(w2d: np.ndarray) -> bytes:
    """Quantise float32 array [N, K] to raw Q8_0 block stream (plain block_q8_0).

    Each row of K elements produces K/32 blocks.
    Block layout: [fp16 d (2 bytes)] [int8 qs[32]] = 34 bytes.

    Matches quantize_row_q8_0_ref:
      d  = amax / 127          (amax = max |x| in the block)
      qi = round(x / d), clipped to [-128, 127]

    Unlike Q4_0 there is no nibble packing and no interleave repack — the plain
    block_q8_0 stream is what nntr_gemm_q8_0_q8_0 / the Q8_0 indirect conv GEMM
    consume directly (weight side, non-interleaved).
    """
    assert w2d.ndim == 2
    N, K = w2d.shape
    assert K % QK8_0 == 0, f"K={K} must be divisible by QK8_0={QK8_0}"

    w2d = w2d.astype(np.float32)
    nblocks_per_row = K // QK8_0
    total_blocks = N * nblocks_per_row

    blocks = w2d.reshape(total_blocks, QK8_0)

    amax = np.max(np.abs(blocks), axis=1)              # (nb,)
    d = (amax / 127.0).astype(np.float32)             # (nb,)
    id_ = np.where(d != 0.0, 1.0 / d, np.float32(0.0))

    # round(x * id), clip to int8 range [-128, 127]
    q = np.clip(np.rint(blocks * id_[:, None]), -128, 127).astype(np.int8)

    # Encode d as fp16 (2 bytes, little-endian)
    d_fp16 = d.astype(np.float16).view(np.uint16)
    d_bytes = d_fp16.view(np.uint8).reshape(total_blocks, 2)

    # [2 bytes d][32 bytes qs] per block
    raw = np.concatenate([d_bytes, q.view(np.uint8)], axis=1).astype(np.uint8)
    return raw.tobytes()  # (nb, 34)


# ---------------------------------------------------------------------------
# Q4_0 repack (matches nntr_make_block_q4_0x4 / nntr_make_block_q4_0x8
#              in q4_0_utils.cpp, ported from gguf_to_nntrainer.py)
# ---------------------------------------------------------------------------

def repack_q4_0(raw_q4_0: bytes, N: int, K: int, interleave: int) -> bytes:
    """Repack plain Q4_0 blocks (N rows x K cols) to nntrainer's q4_0x4/x8 layout.

    Source: block_q4_0 array [N, K/32, 18 bytes].
    Target (interleave=4): block_q4_0x4 groups of 4 rows.
      Per super-block: d[4]*2 bytes, qs[4*16] bytes (XOR'd 0x88).
      Layout of qs: 8 iterations over (nblocks*8) bytes, rearranging by
      row=(i%4) and off=(i//4)*8.
    Target (interleave=8): block_q4_0x8 groups of 8 rows.
      Per super-block: d[8]*2 bytes, qs[8*16] bytes (XOR'd 0x88).

    XOR mask 0x88: negates the offset-by-8 encoding (8 -> 0, so the GEMM
    kernel sees signed nibbles without re-adding the bias).

    Matches nntr_make_block_q4_0x4 / nntr_make_block_q4_0x8 in q4_0_utils.cpp.
    Ported from Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py.
    """
    assert interleave in (4, 8), f"interleave must be 4 or 8, got {interleave}"
    assert N % interleave == 0, \
        f"N={N} must be divisible by interleave={interleave}"
    assert K % QK4_0 == 0
    nblocks = K // QK4_0

    src = np.frombuffer(raw_q4_0, dtype=np.uint8).reshape(N, nblocks, Q4_0_BLOCK_BYTES)
    d_all  = src[:, :, :2]   # (N, nblocks, 2)
    qs_all = src[:, :, 2:]   # (N, nblocks, 16)

    if interleave == 8:
        # block_q4_0x8: d[8] (16 bytes) + qs[128 bytes] per nblocks super-block
        # nntr_make_block_q4_0x8: 16 iterations, row=i%8, off=(i//8)*8
        out = np.empty((N // 8, nblocks, 16 + 128), dtype=np.uint8)
        for g in range(N // 8):
            rows = slice(g * 8, g * 8 + 8)
            # d: 8 rows * 2 bytes -> (nblocks, 16)
            out[g, :, :16] = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 16)
            # qs: (8, nblocks, 16) -> rearrange by row=i%8, off=(i//8)*8
            qs_chunk = qs_all[rows]             # (8, nblocks, 16)
            dst = np.empty((nblocks, 16, 8), dtype=np.uint8)
            for i in range(16):
                row = i % 8
                off = (i // 8) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 128)
            dst ^= np.uint8(0x88)
            out[g, :, 16:] = dst
        return out.tobytes()
    else:  # interleave == 4
        # block_q4_0x4: d[4] (8 bytes) + qs[64 bytes] per nblocks super-block
        # nntr_make_block_q4_0x4: 8 iterations, row=i%4, off=(i//4)*8
        out = np.empty((N // 4, nblocks, 8 + 64), dtype=np.uint8)
        for g in range(N // 4):
            rows = slice(g * 4, g * 4 + 4)
            # d: 4 rows * 2 bytes -> (nblocks, 8)
            out[g, :, :8] = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 8)
            # qs: (4, nblocks, 16) -> rearrange by row=i%4, off=(i//4)*8
            qs_chunk = qs_all[rows]             # (4, nblocks, 16)
            dst = np.empty((nblocks, 8, 8), dtype=np.uint8)
            for i in range(8):
                row = i % 4
                off = (i // 4) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 64)
            dst ^= np.uint8(0x88)
            out[g, :, 8:] = dst
        return out.tobytes()


def repack_q8_0(raw_q8_0: bytes, N: int, K: int) -> bytes:
    """Repack plain Q8_0 blocks (N rows x K cols) to nntrainer's q8_0x4 layout.

    Source: block_q8_0 array [N, K/32, 34 bytes] = [2B fp16 d][32B int8 qs].
    Target: block_q8_0x4 super-blocks of 4 rows (34*4 = 136 bytes each):
      - d[4]  : 8 bytes, the fp16 scales of the 4 rows (this block).
      - qs[128]: for sub in 0..3 (each an 8-col k-subgroup), for row in 0..3,
                 for c in 0..7:  qs[32*sub + row*8 + c] = q[row][32*b + 8*sub + c]

    This is byte-for-byte the activation layout produced by
    __ggml_quantize_mat_q8_0_4x8, so the SMMLA kernel
    nntr_gemm_q8_0_q8_0_4x4_fp16 reads both operands with single contiguous
    vld1q_s8 loads. Total size is unchanged from the plain stream (136*nb = 4*34*nb
    per 4-row group), so the runtime loader/mmap sizing is unaffected -- only the
    intra-tensor byte order changes (offline, zero runtime cost).
    """
    assert N % 4 == 0, f"N={N} must be divisible by 4 for q8_0x4"
    assert K % QK8_0 == 0
    nblocks = K // QK8_0

    src = np.frombuffer(raw_q8_0, dtype=np.uint8).reshape(N, nblocks,
                                                          Q8_0_BLOCK_BYTES)
    d_all = src[:, :, :2]                       # (N, nblocks, 2)
    qs_all = src[:, :, 2:].view(np.int8)        # (N, nblocks, 32)

    out = np.empty((N // 4, nblocks, 8 + 128), dtype=np.uint8)
    for g in range(N // 4):
        rows = slice(g * 4, g * 4 + 4)
        # d: 4 rows * 2 bytes -> (nblocks, 8)
        out[g, :, :8] = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 8)
        # qs: (4, nblocks, 32) -> qs[32*sub + row*8 + c]
        qs_chunk = qs_all[rows]                 # (4, nblocks, 32)
        dst = np.empty((nblocks, 128), dtype=np.int8)
        for sub in range(4):
            for row in range(4):
                dst[:, 32 * sub + row * 8: 32 * sub + row * 8 + 8] = \
                    qs_chunk[row, :, 8 * sub: 8 * sub + 8]
        out[g, :, 8:] = dst.view(np.uint8)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Filter selection
# ---------------------------------------------------------------------------

def is_depthwise(name: str) -> bool:
    """Return True if this tensor is a depthwise (groups > 1) conv filter."""
    return "dw:" in name


def check_conv_filter_eligibility(name: str, shape: list,
                                  allow_larger_kernels: bool,
                                  interleave: int):
    """Check eligibility and return (eligible, reason_string_or_None).

    Applies to any 4-D filter tensor that ends with ':filter'.
    Returns:
      (True, None)          -> quantize
      (False, reason_str)   -> keep FP32, reason_str explains why
    """
    if not name.endswith(":filter"):
        return False, "not a filter"
    if len(shape) != 4:
        return False, "not 4D"

    out_ch, in_ch, kh, kw = shape

    # Depthwise (groups > 1) — always exclude
    if is_depthwise(name):
        return False, "depthwise"

    # Degenerate single-output channel
    if out_ch == 1:
        return False, "out_ch=1"

    # Kernel-size gate: default mode only handles 1x1
    if kh != 1 or kw != 1:
        if not allow_larger_kernels:
            return False, f"kh={kh},kw={kw} (use --all to include)"
        # allow_larger_kernels=True: accept any groups=1 conv

    CRS = in_ch * kh * kw

    # Q4_0 block alignment: both K and N dimensions must be divisible by 32
    reasons = []
    if CRS % QK4_0 != 0:
        reasons.append(f"CRS={CRS} not div32")
    if out_ch % QK4_0 != 0:
        reasons.append(f"out_ch={out_ch} not div32")
    if reasons:
        return False, ", ".join(reasons)

    # Repack alignment: out_ch (N) must be divisible by interleave
    if out_ch % interleave != 0:
        return False, f"out_ch={out_ch} not div interleave={interleave}"

    return True, None


# ---------------------------------------------------------------------------
# Quantize a single conv filter tensor
# ---------------------------------------------------------------------------

def quantize_filter(raw: bytes, shape: list, interleave: int):
    """Quantize a single FP32 conv filter to Q4_0 + repack.

    shape: [out_ch, in_ch, kh, kw]
    Returns: (repacked_bytes, nntr_shape) where nntr_shape = [1, 1, CRS, out_ch].

    For 1x1 (kh=kw=1): CRS = in_ch, identical to original behaviour.
    For larger kernels: CRS = in_ch * kh * kw. The FP32 filter is stored in
    safetensors as [out_ch, in_ch, kh, kw] in C-contiguous (row-major) order.
    reshape(out_ch, -1) flattens the last three axes in order [in_ch, kh, kw],
    which matches nntrainer's im2col column ordering:
      K-index = ic * kh * kw + ki * kw + kj
    giving [N=out_ch, K=CRS] = the matrix that quantize_q4_0 expects.
    """
    out_ch, in_ch, kh, kw = shape
    CRS = in_ch * kh * kw

    w = np.frombuffer(raw, dtype=np.float32).reshape(shape)

    # Flatten to [N=out_ch, K=CRS]: C-contiguous reshape preserves
    # (in_ch, kh, kw) -> K ordering = ic*kh*kw + ki*kw + kj (im2col match)
    w2d = w.reshape(out_ch, CRS)  # [N, K]

    # Step 1: quantize to plain Q4_0 blocks
    raw_q40 = quantize_q4_0(w2d)

    # Step 2: repack to q4_0x{interleave}
    repacked = repack_q4_0(raw_q40, out_ch, CRS, interleave)

    # Verify size
    expected_bytes = (out_ch * CRS // QK4_0) * Q4_0_BLOCK_BYTES
    assert len(repacked) == expected_bytes, \
        f"repacked size {len(repacked)} != expected {expected_bytes}"

    # nntr_shape: [1, 1, K=CRS, N=out_ch]
    nntr_shape = [1, 1, CRS, out_ch]
    return repacked, nntr_shape


def quantize_filter_q8_0(raw: bytes, shape: list):
    """Quantize a single FP32 conv filter to interleaved q8_0x4 Q8_0 blocks.

    shape: [out_ch, in_ch, kh, kw]
    Returns: (q8_bytes, nntr_shape) with nntr_shape = [1, 1, CRS, out_ch].

    Flattening matches quantize_filter (im2col K-order ic*kh*kw + ki*kw + kj).
    The Q8_0 weight is quantized to plain block_q8_0 then repacked offline to the
    q8_0x4 interleaved layout (repack_q8_0) that nntr_gemm_q8_0_q8_0_4x4_fp16
    consumes with contiguous SMMLA loads. Total byte size is identical to the
    plain stream, so the runtime tensor sizing is unchanged.
    """
    out_ch, in_ch, kh, kw = shape
    CRS = in_ch * kh * kw

    w = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    w2d = w.reshape(out_ch, CRS)  # [N=out_ch, K=CRS]

    q8 = quantize_q8_0(w2d)

    expected_bytes = (out_ch * CRS // QK8_0) * Q8_0_BLOCK_BYTES
    assert len(q8) == expected_bytes, \
        f"q8_0 size {len(q8)} != expected {expected_bytes}"

    q8 = repack_q8_0(q8, out_ch, CRS)  # -> q8_0x4 interleaved (same total bytes)
    assert len(q8) == expected_bytes, \
        f"q8_0x4 size {len(q8)} != expected {expected_bytes}"

    nntr_shape = [1, 1, CRS, out_ch]
    return q8, nntr_shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Input FP32 safetensors (yolov11m_fused.safetensors)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: <input stem>_q40.safetensors "
                         "or <input stem>_q40_all.safetensors with --all)")
    ap.add_argument("--target", choices=["arm", "x86"], default="arm",
                    help="Repack interleave target: arm=q4_0x4 (4), x86=q4_0x8 (8). "
                         "Default: arm (for device inference on S23 etc.). "
                         "Ignored for --dtype q8_0 (plain, non-interleaved).")
    ap.add_argument("--dtype", choices=["q4_0", "q8_0"], default="q4_0",
                    help="Weight quantization: q4_0 (4-bit, repacked) or q8_0 "
                         "(8-bit, q8_0x4 interleaved, higher accuracy). Default q4_0.")
    ap.add_argument("--all", dest="all_kernels", action="store_true",
                    help="Also quantize 3x3 and other groups=1 conv filters "
                         "(not just 1x1). Output: yolov11m_fused_q40_all.safetensors.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List tensors that would be quantized without writing")
    args = ap.parse_args()

    is_q8 = args.dtype == "q8_0"
    # Q8_0 is repacked offline to q8_0x4 (4-row interleave) inside
    # quantize_filter_q8_0; out_ch%32 alignment (checked below) already implies
    # the %4 the repack needs, so interleave=1 here keeps the eligibility check a
    # no-op without loosening any alignment constraint.
    interleave = 1 if is_q8 else (4 if args.target == "arm" else 8)

    if args.output is None:
        stem = args.input
        if stem.endswith(".safetensors"):
            stem = stem[:-len(".safetensors")]
        tag = "q80" if is_q8 else "q40"
        suffix = f"_{tag}_all.safetensors" if args.all_kernels else f"_{tag}.safetensors"
        args.output = stem + suffix

    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    if is_q8:
        print(f"Dtype:      q8_0 (q8_0x4 interleaved, 34B/row, 4-row super-blocks)")
    else:
        print(f"Target:     {args.target} (q4_0x{interleave})")
    print(f"Mode:       {'all groups=1 conv (1x1 + larger)' if args.all_kernels else '1x1 only (default)'}")

    header, data, _ = parse_safetensors(args.input)
    tensor_names = [n for n in header if n != "__metadata__"]

    meta = header.get("__metadata__", {})
    print(f"Metadata:   {meta}")

    out_tensors = {}
    n_quantized = 0
    n_skipped_alignment = 0
    n_kept = 0
    total_fp32_bytes = 0
    total_q40_bytes = 0
    skipped_list = []

    # Counters by kernel size for summary
    n_quant_1x1 = 0
    n_quant_larger = 0

    for name in tensor_names:
        entry = header[name]
        dtype = entry["dtype"]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        raw = data[start:end]

        # Pass-through: non-FP32, non-filter, or not 4D
        if dtype != "F32" or not name.endswith(":filter") or len(shape) != 4:
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_kept += 1
            print(f"  [keep]       {name}  {shape}  {dtype}")
            continue

        # Depthwise: always keep FP32
        if is_depthwise(name):
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_kept += 1
            print(f"  [keep-dw]    {name}  {shape}  {dtype}")
            continue

        eligible, reason = check_conv_filter_eligibility(
            name, shape, allow_larger_kernels=args.all_kernels,
            interleave=interleave)

        out_ch, in_ch, kh, kw = shape
        CRS = in_ch * kh * kw
        fp32_bytes = len(raw)

        if not eligible:
            # Determine tag for reporting
            if reason == f"kh={kh},kw={kw} (use --all to include)":
                tag = "[keep-ksize]"
            elif "not div" in reason or "not div interleave" in reason:
                tag = "[skip-align]"
                n_skipped_alignment += 1
                skipped_list.append((name, shape, reason))
            elif reason in ("depthwise", "out_ch=1"):
                tag = "[keep-excl] "
            else:
                tag = "[keep]      "

            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_kept += 1
            print(f"  {tag} {name}  {shape}  ({reason})")
            continue

        # Eligible: quantize
        qtag = "[Q8_0]" if is_q8 else "[Q4_0]"
        print(f"  {qtag}       {name}  {shape}  CRS={CRS}  {fp32_bytes} bytes FP32",
              end="")

        if args.dry_run:
            print(" -> would quantize")
            n_quantized += 1
            if kh == 1 and kw == 1:
                n_quant_1x1 += 1
            else:
                n_quant_larger += 1
            continue

        if is_q8:
            repacked, nntr_shape = quantize_filter_q8_0(raw, shape)
            nntr_dtype = NNTR_DTYPE_Q8_0
        else:
            repacked, nntr_shape = quantize_filter(raw, shape, interleave)
            nntr_dtype = NNTR_DTYPE_Q4_0

        out_tensors[name] = {
            "dtype": SAFETENSORS_DTYPE_Q4_0,   # "U8" (opaque blob)
            "shape": [len(repacked)],            # flat byte count as shape[0]
            "data": repacked,
            "nntr_dtype": nntr_dtype,           # "Q4_0" or "Q8_0"
            "nntr_shape": nntr_shape,            # [1, 1, CRS, out_ch]
        }

        q40_bytes = len(repacked)
        ratio = 100.0 * q40_bytes / fp32_bytes
        print(f" -> {q40_bytes} bytes {args.dtype.upper()} ({ratio:.1f}%)")

        n_quantized += 1
        if kh == 1 and kw == 1:
            n_quant_1x1 += 1
        else:
            n_quant_larger += 1
        total_fp32_bytes += fp32_bytes
        total_q40_bytes += q40_bytes

    print(f"\nSummary:")
    print(f"  Quantized ({args.dtype.upper()}): {n_quantized}"
          f"  (1x1: {n_quant_1x1}, larger: {n_quant_larger})")
    print(f"  Skipped (alignment/interleave): {n_skipped_alignment}")
    if skipped_list:
        for sname, sshape, sreason in skipped_list:
            print(f"    {sname}  {sshape}  ({sreason})")
    print(f"  Kept FP32/other:  {n_kept}")

    if not args.dry_run and n_quantized > 0:
        print(f"\nQuantized conv bytes: {total_fp32_bytes} FP32 -> "
              f"{total_q40_bytes} {args.dtype.upper()} "
              f"({100.0 * total_q40_bytes / total_fp32_bytes:.1f}%)")
        write_safetensors(out_tensors, args.output)
        size = os.path.getsize(args.output)
        in_size = os.path.getsize(args.input)
        print(f"Written: {args.output}")
        print(f"  File size: {size / (1024*1024):.1f} MB  (input: {in_size / (1024*1024):.1f} MB)")

        # Blob layout summary
        if is_q8:
            print(f"\nBlob layout per Q8_0 weight tensor:")
            print(f"  [U8 blob, len = N*CRS/32 * 34 bytes]")
            print(f"  q8_0x4 interleaved: 4-row super-blocks, 136B each (=4*34)")
            print(f"  Per super-block: d[4]*2B (fp16) + qs[128] int8"
                  f" (qs[32*sub+row*8+c])")
            print(f"  Header nntr_shape: [1, 1, K=CRS=in_ch*kh*kw, N=out_ch]")
            print(f"  Header nntr_dtype: Q8_0")
        else:
            print(f"\nBlob layout per Q4_0 weight tensor:")
            print(f"  [U8 blob, len = N*CRS/32 * 18 bytes]")
            print(f"  Repacked as q4_0x{interleave}: groups of {interleave} rows (N=out_ch)")
            print(f"  Per group-of-{interleave}: d[{interleave}]*2 bytes + "
                  f"qs[{interleave}*16] bytes (XOR'd 0x88)")
            print(f"  Header nntr_shape: [1, 1, K=CRS=in_ch*kh*kw, N=out_ch]")
            print(f"  Header nntr_dtype: Q4_0")
    elif not args.dry_run and n_quantized == 0:
        print("\nNothing to quantize.")
    elif args.dry_run:
        print(f"\n(dry-run: no file written)")


if __name__ == "__main__":
    main()
