## @file  gguf_to_nntrainer.py
## @brief Convert a HuggingFace GGUF file of Qwen3 into an nntrainer .bin weight
##        file with Q6_K embedding and Q4_0 fully-connected weights.
##
## The converter reads the quantised tensors directly from the GGUF (no FP32
## round-trip), reorders them to the layout the nntrainer Qwen3 graph expects
## and, for Q4_0 tensors, repacks them into the interleaved q4_0x8 (x86 /
## default) or q4_0x4 (ARM) layout that @c repack_q4_0 produces inside
## nntrainer. Any 1-D RMSNorm weight that is stored as F16 in the GGUF is
## promoted to F32 because nntrainer keeps norm weights in F32.
##
## Expected GGUF tensor quantisation for --strict mode:
##     token_embd.weight                : Q6_K
##     blk.{i}.attn_norm.weight         : F32 or F16
##     blk.{i}.attn_q.weight            : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.attn_q_norm.weight       : F32 or F16
##     blk.{i}.attn_k.weight            : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.attn_k_norm.weight       : F32 or F16
##     blk.{i}.attn_v.weight            : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.attn_output.weight       : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.ffn_norm.weight          : F32 or F16
##     blk.{i}.ffn_up.weight            : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.ffn_gate.weight          : Q4_0 / Q4_1 / Q8_0
##     blk.{i}.ffn_down.weight          : Q4_0 / Q4_1 / Q8_0
##     output_norm.weight               : F32 or F16
##     output.weight (only when not tied): Q6_K (rewritten as Q4_0 or Q6_K)
##
## HuggingFace "Qxxx" GGUFs are commonly mixed: e.g. a Q4_0 GGUF usually has a
## handful of Q4_1 / Q6_K tensors (most often @c ffn_down.weight) alongside the
## Q4_0 majority. This converter therefore transparently dequantises Q4_1 and
## Q8_0 tensors and re-quantises them to Q4_0 even in --strict mode, because
## that is the semantically correct behaviour for these files.
##
## To preserve the precision of the mixed ffn_down tensor, pass
## @c --ffn-down-dtype q6_k . That writes every @c blk.{i}.ffn_down.weight as
## Q6_K instead of Q4_0 (no repack needed). Since all FC layers share
## @c fc_layer_dtype in nntrainer's CausalLM, the loader side must also be set
## to Q6_K for FC weights when using this option.
##
## When --strict is not set, *any* supported GGUF dtype (F32/F16/Q4_0/Q4_1/
## Q8_0/Q6_K) is accepted for FC weights and re-quantised to Q4_0.
##
## @author Claude (for jijoongmoon/nntrainer)

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# GGUF file format
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF"

# GGUF value types
GGUF_U8, GGUF_I8 = 0, 1
GGUF_U16, GGUF_I16 = 2, 3
GGUF_U32, GGUF_I32 = 4, 5
GGUF_F32 = 6
GGUF_BOOL = 7
GGUF_STR = 8
GGUF_ARR = 9
GGUF_U64, GGUF_I64 = 10, 11
GGUF_F64 = 12

# GGML tensor types that this script knows how to interpret.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q8_0 = 8
GGML_Q6_K = 14

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
}

QK4_0 = 32
Q4_0_BLOCK_BYTES = 18         # uint16 d + 16 bytes qs
Q4_1_BLOCK_BYTES = 20         # uint16 d + uint16 m + 16 bytes qs
Q8_0_BLOCK_BYTES = 34         # uint16 d + 32 bytes int8
QK_K = 256
Q6_K_BLOCK_BYTES = 210        # 128 ql + 64 qh + 16 scales + uint16 d

# FC-weight dtypes that we know how to dequantise to fp32 and re-quantise to
# nntrainer's Q4_0. "Soft" matches: tensor is accepted even under --strict.
FC_Q4_0_COMPATIBLE = {GGML_Q4_0, GGML_Q4_1, GGML_Q8_0}


class GGUFReader:
    """Minimal GGUF v2 / v3 reader."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()

    def close(self):
        self.f.close()

    # ------------------------------------------------------------------ io
    def _u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def _read_string(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, typ):
        if typ == GGUF_U8:  return struct.unpack("<B", self.f.read(1))[0]
        if typ == GGUF_I8:  return struct.unpack("<b", self.f.read(1))[0]
        if typ == GGUF_U16: return struct.unpack("<H", self.f.read(2))[0]
        if typ == GGUF_I16: return struct.unpack("<h", self.f.read(2))[0]
        if typ == GGUF_U32: return struct.unpack("<I", self.f.read(4))[0]
        if typ == GGUF_I32: return struct.unpack("<i", self.f.read(4))[0]
        if typ == GGUF_F32: return struct.unpack("<f", self.f.read(4))[0]
        if typ == GGUF_BOOL: return struct.unpack("<B", self.f.read(1))[0] != 0
        if typ == GGUF_STR: return self._read_string()
        if typ == GGUF_ARR:
            arr_typ = self._u32()
            n = self._u64()
            return [self._read_value(arr_typ) for _ in range(n)]
        if typ == GGUF_U64: return struct.unpack("<Q", self.f.read(8))[0]
        if typ == GGUF_I64: return struct.unpack("<q", self.f.read(8))[0]
        if typ == GGUF_F64: return struct.unpack("<d", self.f.read(8))[0]
        raise ValueError(f"unknown GGUF metadata type: {typ}")

    # --------------------------------------------------------------- parse
    def _read_header(self):
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"not a GGUF file (magic=0x{magic:08x}): {self.path}")
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()

    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.n_kv):
            key = self._read_string()
            typ = self._u32()
            self.metadata[key] = self._read_value(typ)

    def _read_tensor_info(self):
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dim = self._u32()
            dims = [self._u64() for _ in range(n_dim)]
            # GGUF stores dims in reverse (ggml order: innermost first).
            # We present shape in numpy order: outer first.
            shape = tuple(reversed(dims))
            typ = self._u32()
            offset = self._u64()
            self.tensors[name] = {
                "shape": shape,
                "type": typ,
                "offset": offset,
            }
        alignment = self.metadata.get("general.alignment", 32)
        pos = self.f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        self.data_start = pos + pad

    # --------------------------------------------------------------- data
    def tensor_bytes_size(self, info):
        numel = 1
        for d in info["shape"]:
            numel *= d
        t = info["type"]
        if t == GGML_F32:   return numel * 4
        if t == GGML_F16:   return numel * 2
        if t == GGML_Q4_0:
            assert numel % QK4_0 == 0, "Q4_0 element count must divide 32"
            return (numel // QK4_0) * Q4_0_BLOCK_BYTES
        if t == GGML_Q4_1:
            assert numel % QK4_0 == 0, "Q4_1 element count must divide 32"
            return (numel // QK4_0) * Q4_1_BLOCK_BYTES
        if t == GGML_Q8_0:
            assert numel % 32 == 0, "Q8_0 element count must divide 32"
            return (numel // 32) * Q8_0_BLOCK_BYTES
        if t == GGML_Q6_K:
            assert numel % QK_K == 0, "Q6_K element count must divide 256"
            return (numel // QK_K) * Q6_K_BLOCK_BYTES
        raise ValueError(
            f"unsupported GGML type {t} "
            f"({GGML_TYPE_NAMES.get(t, '?')}) in converter")

    def read_tensor_raw(self, name):
        info = self.tensors[name]
        size = self.tensor_bytes_size(info)
        self.f.seek(self.data_start + info["offset"])
        return self.f.read(size), info


# ---------------------------------------------------------------------------
# Dequant / quant helpers (kept minimal; used when GGUF dtype does not match)
# ---------------------------------------------------------------------------
def dequant_q4_0(buf: bytes, numel: int) -> np.ndarray:
    """Dequantise a Q4_0 byte buffer into fp32."""
    nb = numel // QK4_0
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q4_0_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 2:]  # (nb, 16) nibbles
    low = (q & 0x0F).astype(np.int32) - 8
    high = (q >> 4).astype(np.int32) - 8
    # Layout: for i in [0, 16): q[i] low -> elem i, q[i] high -> elem i+16
    out = np.concatenate([low, high], axis=1).astype(np.float32) * d
    return out.reshape(numel)


def dequant_q4_1(buf: bytes, numel: int) -> np.ndarray:
    """Dequantise a Q4_1 byte buffer into fp32. x = d*q + m, q in [0,15]."""
    nb = numel // QK4_0
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q4_1_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    m = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 4:]  # (nb, 16) nibbles
    low = (q & 0x0F).astype(np.float32)
    high = (q >> 4).astype(np.float32)
    out = np.concatenate([low, high], axis=1) * d + m
    return out.reshape(numel)


def dequant_q8_0(buf: bytes, numel: int) -> np.ndarray:
    """Dequantise a Q8_0 byte buffer into fp32."""
    nb = numel // 32
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nb, Q8_0_BLOCK_BYTES)
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    q = arr[:, 2:].copy().view(np.int8).reshape(nb, 32).astype(np.float32)
    return (q * d).reshape(numel)


def dequant_q6_k(buf: bytes, numel: int) -> np.ndarray:
    """Dequantise a Q6_K byte buffer into fp32.
    Port of ggml-quants.c dequantize_row_q6_K.
    """
    nb = numel // QK_K
    out = np.empty((nb, QK_K), dtype=np.float32)
    view = memoryview(buf)
    for b in range(nb):
        base = b * Q6_K_BLOCK_BYTES
        ql_all = np.frombuffer(view[base:base + 128], dtype=np.uint8)
        qh_all = np.frombuffer(view[base + 128:base + 192], dtype=np.uint8)
        sc_all = np.frombuffer(view[base + 192:base + 208], dtype=np.int8)
        d = np.frombuffer(
            view[base + 208:base + 210], dtype=np.float16).astype(np.float32)[0]
        y = np.empty(QK_K, dtype=np.float32)
        for half in range(2):                         # two 128-wide halves
            ql = ql_all[half * 64:(half + 1) * 64]
            qh = qh_all[half * 32:(half + 1) * 32]
            sc = sc_all[half * 8:(half + 1) * 8]
            y_off = half * 128
            for l in range(32):
                is_ = l // 16
                q1 = int(ql[l + 0] & 0xF) | (((int(qh[l]) >> 0) & 3) << 4)
                q2 = int(ql[l + 32] & 0xF) | (((int(qh[l]) >> 2) & 3) << 4)
                q3 = int(ql[l + 0] >> 4) | (((int(qh[l]) >> 4) & 3) << 4)
                q4 = int(ql[l + 32] >> 4) | (((int(qh[l]) >> 6) & 3) << 4)
                y[y_off + l + 0]  = d * sc[is_ + 0] * (q1 - 32)
                y[y_off + l + 32] = d * sc[is_ + 2] * (q2 - 32)
                y[y_off + l + 64] = d * sc[is_ + 4] * (q3 - 32)
                y[y_off + l + 96] = d * sc[is_ + 6] * (q4 - 32)
        out[b] = y
    return out.reshape(numel)


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Quantise fp32 row-major (..., K) -> raw Q4_0 block stream."""
    flat = x.reshape(-1, QK4_0).astype(np.float32)
    nb = flat.shape[0]
    amax_idx = np.argmax(np.abs(flat), axis=1)
    amax = flat[np.arange(nb), amax_idx]
    d = amax / -8.0
    id_ = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
    # Quantise to [-8, 7], then store +8 to fit in 4 bits unsigned.
    q = np.clip(
        np.rint(flat * id_[:, None]) + 8.0, 0, 15).astype(np.uint8)
    low = q[:, :16]
    high = q[:, 16:]
    qs = (low | (high << 4)).astype(np.uint8)        # (nb, 16)
    d_fp16 = d.astype(np.float16).view(np.uint16).reshape(nb, 1)
    # Interleave: 2 bytes d, 16 bytes qs
    d_bytes = d_fp16.view(np.uint8).reshape(nb, 2)
    blocks = np.concatenate([d_bytes, qs], axis=1).astype(np.uint8)
    return blocks.tobytes()


def quantize_q6_k(x: np.ndarray) -> bytes:
    """Quantise fp32 (..., K) -> raw Q6_K block stream.

    This is a straightforward port of ggml-quants.c's quantize_row_q6_K_ref
    (no imatrix). It produces bytes identical in layout to the GGUF block.
    """
    flat = x.reshape(-1, QK_K).astype(np.float32)
    nb = flat.shape[0]
    out = bytearray(nb * Q6_K_BLOCK_BYTES)
    L = np.empty(QK_K, dtype=np.int8)
    scales = np.empty(QK_K // 16, dtype=np.float32)
    for b in range(nb):
        blk = flat[b]
        # Step 1: per 16-element sub-block, find scale.
        max_scale = np.float32(0.0)
        max_abs_scale = np.float32(0.0)
        for ib in range(QK_K // 16):
            sub = blk[ib * 16:(ib + 1) * 16]
            amax_idx = int(np.argmax(np.abs(sub)))
            amax = float(sub[amax_idx])
            if amax == 0.0:
                scales[ib] = 0.0
                continue
            s = amax / -32.0
            scales[ib] = s
            if abs(s) > max_abs_scale:
                max_abs_scale = np.float32(abs(s))
                max_scale = np.float32(s)
        if max_abs_scale == 0.0:
            # all zeros
            base = b * Q6_K_BLOCK_BYTES
            out[base:base + Q6_K_BLOCK_BYTES] = bytes(Q6_K_BLOCK_BYTES)
            continue
        iscale = np.float32(-128.0 / max_scale)
        d_super = np.float32(1.0 / iscale)   # super-block scale (stored as fp16)
        int8_scales = np.empty(16, dtype=np.int8)
        for ib in range(16):
            l_scale = int(round(iscale * scales[ib]))
            l_scale = max(-128, min(127, l_scale))
            int8_scales[ib] = l_scale
        # Step 2: quantise each element using sub-block scale.
        for ib in range(16):
            d_sub = d_super * np.float32(int8_scales[ib])
            sub = blk[ib * 16:(ib + 1) * 16]
            if d_sub == 0.0:
                L[ib * 16:(ib + 1) * 16] = 0
                continue
            inv = np.float32(1.0 / d_sub)
            q = np.rint(sub * inv).astype(np.int32) + 32
            q = np.clip(q, 0, 63)
            L[ib * 16:(ib + 1) * 16] = q.astype(np.int8)
        # Step 3: pack into ql (low nibble) and qh (upper 2 bits).
        ql = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(64, dtype=np.uint8)
        Lu = L.view(np.uint8)
        for half in range(2):
            y_off = half * 128
            ql_off = half * 64
            qh_off = half * 32
            for l in range(32):
                q1 = int(Lu[y_off + l + 0])
                q2 = int(Lu[y_off + l + 32])
                q3 = int(Lu[y_off + l + 64])
                q4 = int(Lu[y_off + l + 96])
                ql[ql_off + l + 0]  = (q1 & 0xF) | ((q3 & 0xF) << 4)
                ql[ql_off + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
                qh[qh_off + l] = (
                    (((q1 >> 4) & 3) << 0) |
                    (((q2 >> 4) & 3) << 2) |
                    (((q3 >> 4) & 3) << 4) |
                    (((q4 >> 4) & 3) << 6)
                )
        base = b * Q6_K_BLOCK_BYTES
        out[base:base + 128]          = bytes(ql)
        out[base + 128:base + 192]    = bytes(qh)
        out[base + 192:base + 208]    = int8_scales.tobytes()
        out[base + 208:base + 210]    = np.float16(d_super).tobytes()
    return bytes(out)


# ---------------------------------------------------------------------------
# Q4_0 repack into q4_0x8 / q4_0x4 (matches nntrainer/.../nntr_ggml_impl_*)
# ---------------------------------------------------------------------------
def _xor_mask_bytes(n: int) -> np.ndarray:
    return np.full(n, 0x88, dtype=np.uint8)


def repack_q4_0(raw_q4_0: bytes, N: int, K: int, interleave: int) -> bytes:
    """Repack raw GGUF Q4_0 blocks (N rows × K cols) into nntrainer's
    q4_0x8 / q4_0x4 layout with the 0x88 XOR mask applied to the nibbles.

    Source layout: block_q4_0 array of shape (N, K/32), each 18 bytes.
    Target layout:
        x8: groups of 8 rows -> block_q4_0x8 (16 bytes d + 128 bytes qs) * (K/32)
        x4: groups of 4 rows -> block_q4_0x4 (8 bytes d + 64 bytes qs) * (K/32)
    """
    assert interleave in (4, 8)
    assert N % interleave == 0, \
        f"N={N} must be divisible by interleave group size {interleave}"
    assert K % QK4_0 == 0
    nblocks = K // QK4_0

    src = np.frombuffer(raw_q4_0, dtype=np.uint8).reshape(N, nblocks, Q4_0_BLOCK_BYTES)
    d_all = src[:, :, :2]     # (N, nblocks, 2)
    qs_all = src[:, :, 2:]    # (N, nblocks, 16)

    if interleave == 8:
        # Per nntr_ggml_impl_fallback.cpp::nntr_make_block_q4_0x8:
        #   end = QK4_0 * 4 / 8 = 16 iterations per super-block,
        #   src_id = i % 8, src_offset = (i/8) * 8, dst_offset = i * 8.
        # I.e. 8 bytes from each of the 8 rows (low half), then 8 bytes from
        # each row (high half) — all XOR'd with 0x88.
        out_super = np.empty((N // 8, nblocks, 16 + 128), dtype=np.uint8)
        for g in range(N // 8):
            rows = slice(g * 8, g * 8 + 8)
            # d: 8 * 2 bytes in row order.
            d_chunk = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 16)
            out_super[g, :, :16] = d_chunk
            # qs: for each of 16 positions i, take 8 bytes from row (i%8)
            # starting at offset (i//8)*8.
            qs_chunk = qs_all[rows]                        # (8, nblocks, 16)
            # Rearrange to (nblocks, 16 positions, 8 bytes)
            # position i -> row=i%8, src_off=(i//8)*8
            dst = np.empty((nblocks, 16, 8), dtype=np.uint8)
            for i in range(16):
                row = i % 8
                off = (i // 8) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 128)
            dst ^= 0x88
            out_super[g, :, 16:] = dst
        return out_super.tobytes()
    else:  # interleave == 4
        # Per nntr_ggml_impl_fallback.cpp::nntr_make_block_q4_0x4 with
        # blck_size_interleave==8 (that is how ggml_interface.cpp calls it):
        #   end = QK4_0 * 2 / 8 = 8 iterations,
        #   src_id = i % 4, src_offset = (i/4) * 8, dst_offset = i * 8.
        out_super = np.empty((N // 4, nblocks, 8 + 64), dtype=np.uint8)
        for g in range(N // 4):
            rows = slice(g * 4, g * 4 + 4)
            d_chunk = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 8)
            out_super[g, :, :8] = d_chunk
            qs_chunk = qs_all[rows]                        # (4, nblocks, 16)
            dst = np.empty((nblocks, 8, 8), dtype=np.uint8)
            for i in range(8):
                row = i % 4
                off = (i // 4) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 64)
            dst ^= 0x88
            out_super[g, :, 8:] = dst
        return out_super.tobytes()


# ---------------------------------------------------------------------------
# GGUF tensor -> nntrainer-shaped bytes
# ---------------------------------------------------------------------------
def _gguf_tensor_to_fp32(reader: GGUFReader, name: str) -> np.ndarray:
    buf, info = reader.read_tensor_raw(name)
    t = info["type"]
    numel = 1
    for d in info["shape"]:
        numel *= d
    if t == GGML_F32:
        arr = np.frombuffer(buf, dtype=np.float32).copy()
    elif t == GGML_F16:
        arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    elif t == GGML_Q4_0:
        arr = dequant_q4_0(buf, numel)
    elif t == GGML_Q4_1:
        arr = dequant_q4_1(buf, numel)
    elif t == GGML_Q8_0:
        arr = dequant_q8_0(buf, numel)
    elif t == GGML_Q6_K:
        arr = dequant_q6_k(buf, numel)
    else:
        raise ValueError(
            f"dequant for GGML type {t} ({GGML_TYPE_NAMES.get(t,'?')}) "
            f"not implemented; tensor={name}")
    return arr.reshape(info["shape"])


def write_norm(out, reader: GGUFReader, name: str, expected_len: int):
    """RMSNorm scale: always FP32 on disk (1D)."""
    arr = _gguf_tensor_to_fp32(reader, name).astype(np.float32).ravel()
    assert arr.size == expected_len, \
        f"{name}: got len {arr.size}, expected {expected_len}"
    out.write(arr.tobytes())


def write_embedding_q6k(out, reader: GGUFReader, name: str,
                         vocab: int, hidden: int, strict: bool):
    info = reader.tensors[name]
    if info["type"] == GGML_Q6_K:
        buf, _ = reader.read_tensor_raw(name)
        assert len(buf) == (vocab * hidden // QK_K) * Q6_K_BLOCK_BYTES
        out.write(buf)
        return
    if strict:
        raise ValueError(
            f"embedding {name} is {GGML_TYPE_NAMES.get(info['type'],'?')} "
            "but Q6_K is required; re-run without --strict to requantise")
    arr = _gguf_tensor_to_fp32(reader, name)
    assert arr.shape == (vocab, hidden), \
        f"embedding shape {arr.shape} != ({vocab},{hidden})"
    out.write(quantize_q6_k(arr.reshape(-1, hidden)))


def write_fc_fp32(out, reader: GGUFReader, name: str,
                   out_features: int, in_features: int):
    """Write a Linear(in, out) weight as plain FP32 (no quantisation).

    Used for ARM-vs-x86 divergence debugging: if a model in Q4_0 loops on ARM
    but the same model in FP32 doesn't, the ARM Q4_0 code path is implicated."""
    arr = _gguf_tensor_to_fp32(reader, name).astype(np.float32, copy=False)
    assert arr.shape == (out_features, in_features), \
        f"{name} shape {arr.shape} != ({out_features},{in_features})"
    info = reader.tensors[name]
    if info["type"] not in (GGML_F32,):
        print(f"  [dequant] {name}: {GGML_TYPE_NAMES.get(info['type'],'?')} -> FP32")
    out.write(arr.tobytes())


def write_fc_q6_k(out, reader: GGUFReader, name: str,
                   out_features: int, in_features: int):
    """Write a Linear(in, out) weight as nntrainer Q6_K.

    Q6_K is row-major, block_q6_K[out_features, in_features/QK_K]. No
    repacking is required — this is the same layout nntrainer's Q6_K tensor
    loads. When the source is already Q6_K we byte-copy; otherwise we
    dequantise and re-quantise (lossy but much better than Q4_0)."""
    if in_features % QK_K != 0:
        raise ValueError(
            f"{name}: in_features={in_features} must be divisible by "
            f"QK_K={QK_K} to write as Q6_K")
    info = reader.tensors[name]
    t = info["type"]
    if t == GGML_Q6_K:
        raw, _ = reader.read_tensor_raw(name)
        expected = (out_features * in_features // QK_K) * Q6_K_BLOCK_BYTES
        assert len(raw) == expected, \
            f"{name}: raw Q6_K size {len(raw)} != expected {expected}"
        out.write(raw)
        return
    print(f"  [requant] {name}: {GGML_TYPE_NAMES.get(t,'?')} -> Q6_K")
    arr = _gguf_tensor_to_fp32(reader, name)
    assert arr.shape == (out_features, in_features), \
        f"{name} shape {arr.shape} != ({out_features},{in_features})"
    out.write(quantize_q6_k(arr.reshape(-1, in_features)))


def write_fc_q4_0(out, reader: GGUFReader, name: str,
                   out_features: int, in_features: int,
                   interleave: int, strict: bool):
    """Write a Linear(in, out) weight as nntrainer Q4_0 (repacked)."""
    if in_features % QK4_0 != 0:
        raise ValueError(
            f"{name}: in_features={in_features} must be divisible by {QK4_0}")
    if out_features % interleave != 0:
        raise ValueError(
            f"{name}: out_features={out_features} must be divisible by "
            f"interleave={interleave} (required by nntrainer repack_q4_0)")
    info = reader.tensors[name]
    t = info["type"]
    if t == GGML_Q4_0:
        raw, _ = reader.read_tensor_raw(name)
        expected = (out_features * in_features // QK4_0) * Q4_0_BLOCK_BYTES
        assert len(raw) == expected, \
            f"{name}: raw Q4_0 size {len(raw)} != expected {expected}"
    elif t in FC_Q4_0_COMPATIBLE:
        # Q4_1 / Q8_0: common mixed-quant variants in a "Q4_0" GGUF.
        # Always dequantise and re-quantise to Q4_0 — this is semantically
        # what the user asked for even under --strict.
        print(f"  [requant] {name}: {GGML_TYPE_NAMES.get(t,'?')} -> Q4_0")
        arr = _gguf_tensor_to_fp32(reader, name)
        assert arr.shape == (out_features, in_features), \
            f"{name} shape {arr.shape} != ({out_features},{in_features})"
        raw = quantize_q4_0(arr)
    else:
        if strict:
            raise ValueError(
                f"{name} is {GGML_TYPE_NAMES.get(t,'?')} but Q4_0 (or a "
                "Q4_0-compatible type like Q4_1/Q8_0) is required; "
                "re-run without --strict to requantise")
        arr = _gguf_tensor_to_fp32(reader, name)
        assert arr.shape == (out_features, in_features), \
            f"{name} shape {arr.shape} != ({out_features},{in_features})"
        raw = quantize_q4_0(arr)
    out.write(repack_q4_0(raw, out_features, in_features, interleave))


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
QWEN3_TENSORS = {
    "token_embd":    "token_embd.weight",
    "attn_norm":     "blk.{i}.attn_norm.weight",
    "attn_q":        "blk.{i}.attn_q.weight",
    "attn_q_norm":   "blk.{i}.attn_q_norm.weight",
    "attn_k":        "blk.{i}.attn_k.weight",
    "attn_k_norm":   "blk.{i}.attn_k_norm.weight",
    "attn_v":        "blk.{i}.attn_v.weight",
    "attn_output":   "blk.{i}.attn_output.weight",
    "ffn_norm":      "blk.{i}.ffn_norm.weight",
    "ffn_up":        "blk.{i}.ffn_up.weight",
    "ffn_gate":      "blk.{i}.ffn_gate.weight",
    "ffn_down":      "blk.{i}.ffn_down.weight",
    "output_norm":   "output_norm.weight",
    "output":        "output.weight",
}


def convert(args):
    reader = GGUFReader(args.gguf)
    md = reader.metadata

    # --- Pull Qwen3 config from metadata --------------------------------
    arch = md.get("general.architecture", "")
    if arch != "qwen3":
        print(f"[warn] general.architecture='{arch}', expected 'qwen3'")
    n_layers = int(md["qwen3.block_count"])
    hidden   = int(md["qwen3.embedding_length"])
    vocab    = int(md.get("qwen3.vocab_size",
                          len(md.get("tokenizer.ggml.tokens", []))))
    n_heads  = int(md["qwen3.attention.head_count"])
    n_kv     = int(md["qwen3.attention.head_count_kv"])
    head_dim = int(md.get("qwen3.attention.key_length", hidden // n_heads))
    ff_dim   = int(md["qwen3.feed_forward_length"])
    tied = bool(md.get("qwen3.tie_word_embeddings", False)) \
        or ("output.weight" not in reader.tensors)

    q_size  = n_heads * head_dim
    kv_size = n_kv * head_dim

    print("Qwen3 config from GGUF:")
    print(f"  layers     : {n_layers}")
    print(f"  hidden     : {hidden}")
    print(f"  vocab      : {vocab}")
    print(f"  n_heads    : {n_heads}   (kv={n_kv}, head_dim={head_dim})")
    print(f"  ffn        : {ff_dim}")
    print(f"  tied embed : {tied}")
    print(f"  target     : {args.target} (Q4_0 interleave={args.interleave})")
    fc_dtype = args.fc_dtype
    # --ffn-down-dtype q6_k retains old behaviour (ffn_down only → Q6_K);
    # --fc-dtype overrides that for every FC.
    ffn_down_dtype = fc_dtype if fc_dtype != "q4_0" else args.ffn_down_dtype
    print(f"  fc_dtype   : {fc_dtype.upper()}")
    print(f"  ffn_down   : {ffn_down_dtype.upper()}")
    print()

    # --- Write the nntrainer .bin ---------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as out:
        def write_fc(name, N_out, N_in, dtype):
            if dtype == "q4_0":
                write_fc_q4_0(out, reader, name, N_out, N_in,
                              args.interleave, args.strict)
            elif dtype == "q6_k":
                write_fc_q6_k(out, reader, name, N_out, N_in)
            elif dtype == "fp32":
                write_fc_fp32(out, reader, name, N_out, N_in)
            else:
                raise ValueError(f"unknown fc dtype {dtype}")
        # 1. Embedding
        write_embedding_q6k(
            out, reader, QWEN3_TENSORS["token_embd"],
            vocab, hidden, args.strict)

        # 2. Decoder blocks
        for i in range(n_layers):
            def n(k):
                return QWEN3_TENSORS[k].format(i=i)

            write_norm(out, reader, n("attn_norm"), hidden)
            write_fc(n("attn_q"),       q_size,  hidden,  fc_dtype)
            write_norm(out, reader, n("attn_q_norm"), head_dim)
            write_fc(n("attn_k"),       kv_size, hidden,  fc_dtype)
            write_norm(out, reader, n("attn_k_norm"), head_dim)
            write_fc(n("attn_v"),       kv_size, hidden,  fc_dtype)
            write_fc(n("attn_output"),  hidden,  q_size,  fc_dtype)
            write_norm(out, reader, n("ffn_norm"), hidden)
            write_fc(n("ffn_up"),       ff_dim,  hidden,  fc_dtype)
            write_fc(n("ffn_gate"),     ff_dim,  hidden,  fc_dtype)
            write_fc(n("ffn_down"),     hidden,  ff_dim,  ffn_down_dtype)
            print(f"  layer {i:2d}/{n_layers} written")

        # 3. output_norm
        write_norm(out, reader, QWEN3_TENSORS["output_norm"], hidden)

        # 4. lm_head (only if NOT tied) — kept at Q4_0 so nntrainer's
        # lmhead_dtype can stay Q4_0 even when FC is Q6_K/FP32.
        if not tied:
            write_fc_q4_0(out, reader, QWEN3_TENSORS["output"],
                          vocab, hidden, args.interleave, args.strict)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MiB)")

    # --- Drop a matching nntr_config.json next to it --------------------
    if args.emit_nntr_config:
        fc_dtype_cfg = {"q4_0": "Q4_0", "q6_k": "Q6_K", "fp32": "FP32"}[args.fc_dtype]
        cfg = {
            "model_type": "CausalLM",
            "model_tensor_type": f"{fc_dtype_cfg}-FP32",
            "model_file_name": os.path.basename(args.output),
            "fc_layer_dtype": fc_dtype_cfg,
            "embedding_dtype": "Q6_K",
            "lmhead_dtype": "Q6_K" if tied else "Q4_0",
            "lora_rank": 0,
            "lora_alpha": 0,
            "lora_target": [],
            "bad_word_ids": [],
            "fsu": False,
            "fsu_lookahead": 2,
            "num_to_generate": 512,
            "init_seq_len": 1024,
            "max_seq_len": 2048,
            "batch_size": 1,
            "tokenizer_file":
                "/tmp/nntrainer/Applications/CausalLM/res/qwen3-0.6b/tokenizer.json",
            "sample_input":
                "<|im_start|>user\nGive me a short introduction to large "
                "language model.<|im_end|>\n<|im_start|>assistant\n",
        }
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            "nntr_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"Wrote {cfg_path}")

    reader.close()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert a HuggingFace Qwen3 GGUF to nntrainer .bin "
                    "(Q6_K embedding + Q4_0 FC).")
    ap.add_argument("gguf", help="Path to the source .gguf file")
    ap.add_argument("-o", "--output",
                    default="nntr_qwen3_0.6b_q40_embdq6k.bin",
                    help="Output .bin path")
    ap.add_argument("--target", choices=["x86", "arm"], default="x86",
                    help="Target CPU for Q4_0 repack layout "
                         "(x86 -> q4_0x8, arm -> q4_0x4)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if GGUF tensor dtype differs from target dtype")
    ap.add_argument("--fc-dtype", choices=["q4_0", "q6_k", "fp32"],
                    default="q4_0",
                    help="Target dtype for ALL FC weights. 'q6_k' ~= 6-bit, "
                         "'fp32' = no quantisation (diagnostic). When not "
                         "'q4_0', nntr_config.json's fc_layer_dtype is set "
                         "accordingly so the runtime matches.")
    ap.add_argument("--ffn-down-dtype", choices=["q4_0", "q6_k"],
                    default="q4_0",
                    help="Like --fc-dtype but only applied to ffn_down. "
                         "Overridden by --fc-dtype when that is not q4_0. "
                         "(Note: nntrainer currently has a single "
                         "fc_layer_dtype, so mixed Q4_0 FC + Q6_K ffn_down "
                         "will not load cleanly without further changes.)")
    ap.add_argument("--emit-nntr-config", action="store_true",
                    help="Also write nntr_config.json next to the .bin")
    args = ap.parse_args()
    args.interleave = 8 if args.target == "x86" else 4
    return args


if __name__ == "__main__":
    sys.exit(convert(parse_args()) or 0)
