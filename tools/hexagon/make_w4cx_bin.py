#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    make_w4cx_bin.py
# @brief   Build the Qwen3 w4cx (per-channel int4) checkpoint (.bin) from a HuggingFace directory
# @author  dlwlzzero <dlwlzzero@gmail.com>
"""Build the Qwen3 w4cx checkpoint (.bin) straight from a HuggingFace directory.

The w4cx format (issue #65 S1) is the W8_CX stream of make_w8cx_bin.py with
every 2-D tensor whose class is not named in `--i8-tensors` quantised per
output channel to int4 instead of int8: symmetric RTN, scale = absmax / 7,
code = lround(x * 7 / absmax) in [-7, 7] (inside HexKL's [-8, 7]), still
stored one code per byte so the stream keeps the W8 layout and size. The
file starts with a 64-byte header (`W4CX`, version 1, n_layers, the int8
class mask, bits) that `Qwen3W8cxBin` recognises and maps to the
`w4cx_down8` image. The default `--i8-tensors down,embed` is the split the
DSP image supports (down keeps its int16 activation on HVX, embed / LOGITS
are #65 S3); a set without embed and down is refused before anything is
written. Class names, in mask-bit order: embed,q,k,v,o,gate,up,down.

The reader, the ABI v5 `MATMUL_W4A8` lowering and the x86 reference live on
`hvx_impl` (#69); this producer is kept on the `hvx_w4cx` branch until the
W4 direction is decided (per-channel int4 RTN is 2.05x the W8 perplexity on
Qwen3-0.6B, see HEXAGON.md section 8.1). The shared loader, rounding and
writer come from make_w8cx_bin.py next to this file, so a W8 file built by
that script and the int8 tensors of a w4cx file are bit-identical.

Usage:
    make_w4cx_bin.py <hf_dir> <out.bin> [--layers N] [--i8-tensors down,embed]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_w8cx_bin import load_safetensors, lround_half_away, to_f32  # noqa: E402

TENSOR_CLASSES = ["embed", "q", "k", "v", "o", "gate", "up", "down"]
HEADER_BYTES = 64


def quant_cx(w, qmax):
    """w: float32 [N][K] -> (int8 [N][K], float32 [N]); qmax 127 is the W8_CX
    primitive, qmax 7 the int4 variant with the same formula."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    qm = np.float32(qmax)
    amax = np.abs(w).max(axis=1).astype(np.float32)               # float32 max
    scale = (amax / qm).astype(np.float32)                         # float32 division
    inv = np.where(amax > 0, qm / amax, np.float32(0.0)).astype(np.float32)
    y = (w * inv[:, None]).astype(np.float32)                      # float32 multiply
    q = np.clip(lround_half_away(y), -qmax, qmax).astype(np.int8)
    return q, scale


def class_mask(names):
    """'down,embed' -> HexTensorBit mask (bit i = TENSOR_CLASSES[i])."""
    mask = 0
    for name in filter(None, names.split(",")):
        if name not in TENSOR_CLASSES:
            raise SystemExit(f"unknown tensor class '{name}', expected one of {TENSOR_CLASSES}")
        mask |= 1 << TENSOR_CLASSES.index(name)
    return mask


class Writer:
    def __init__(self, fh, i8_mask):
        self.fh = fh
        self.nbytes = 0
        self.i8_mask = i8_mask

    def f32(self, a, n):
        a = np.ascontiguousarray(a, dtype=np.float32)
        assert a.size == n, (a.shape, n)
        self.fh.write(a.tobytes())
        self.nbytes += a.nbytes

    def quantized(self, w, n, k, label, cls):
        assert w.shape == (n, k), (label, w.shape, (n, k))
        i8 = bool(self.i8_mask & (1 << TENSOR_CLASSES.index(cls)))
        q, s = quant_cx(w, 127 if i8 else 7)
        self.fh.write(q.tobytes())
        self.fh.write(s.tobytes())
        self.nbytes += q.nbytes + s.nbytes
        print(f"  {label:44s} [{n}x{k}] {'int8' if i8 else 'int4'} + {n} scales", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir", help="HuggingFace model directory (config.json + model.safetensors)")
    ap.add_argument("out", help="output .bin path")
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (bring-up images)")
    ap.add_argument("--i8-tensors", default="down,embed",
                    help="classes kept int8 (names from %s)" % ",".join(TENSOR_CLASSES))
    args = ap.parse_args()
    i8_mask = class_mask(args.i8_tensors)
    if i8_mask & 0x81 != 0x81:
        # Only the w4cx_down8 image exists (Qwen3W8cxBin::apply_layout); a
        # 4-bit embed or down is #65 S3. Refuse before writing 600 MB.
        raise SystemExit("--i8-tensors must include embed and down until the w4cx "
                         "(4-bit embed / down) layout of #65 S3 exists")

    with open(os.path.join(args.hf_dir, "config.json")) as f:
        cfg = json.load(f)
    n_layers = args.layers or cfg["num_hidden_layers"]
    hidden, ffn, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    n_heads, n_kv, head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    qdim, kvdim = n_heads * head_dim, n_kv * head_dim
    if not cfg.get("tie_word_embeddings", False):
        print("warning: lm_head is not tied; the W8_CX layout has no separate lm_head "
              "(LOGITS reuses the embedding table)", file=sys.stderr)

    st_path = os.path.join(args.hf_dir, "model.safetensors")
    tensors = load_safetensors(st_path)
    T = lambda name: to_f32(tensors[name])

    def qb(n, k):
        return n * k + 4 * n

    expected = (HEADER_BYTES
                + qb(vocab, hidden)
                + n_layers * (4 * hidden + qb(qdim, hidden) + 4 * head_dim + qb(kvdim, hidden)
                              + 4 * head_dim + qb(kvdim, hidden) + qb(hidden, qdim) + 4 * hidden
                              + qb(ffn, hidden) + qb(ffn, hidden) + qb(hidden, ffn))
                + 4 * hidden)

    print(f"hf={args.hf_dir} layers={n_layers} hidden={hidden} ffn={ffn} vocab={vocab} "
          f"heads={n_heads}/{n_kv} head_dim={head_dim} bits=4 "
          f"i8={[c for c in TENSOR_CLASSES if i8_mask >> TENSOR_CLASSES.index(c) & 1]} "
          f"-> {args.out} ({expected:,} bytes)")

    with open(args.out, "wb") as fh:
        # struct Qwen3W4cxBinHeader (qwen3_w8cx_bin.h): magic, version,
        # n_layers, i8_mask, bits, 44 B pad = 64 B
        fh.write(struct.pack("<4sIIII44x", b"W4CX", 1, n_layers, i8_mask, 4))
        w = Writer(fh, i8_mask)
        w.quantized(T("model.embed_tokens.weight"), vocab, hidden, "embed_tokens", "embed")
        for i in range(n_layers):
            p = f"model.layers.{i}."
            w.f32(T(p + "input_layernorm.weight"), hidden)
            w.quantized(T(p + "self_attn.q_proj.weight"), qdim, hidden, f"{i}.q_proj", "q")
            w.f32(T(p + "self_attn.q_norm.weight"), head_dim)
            w.quantized(T(p + "self_attn.k_proj.weight"), kvdim, hidden, f"{i}.k_proj", "k")
            w.f32(T(p + "self_attn.k_norm.weight"), head_dim)
            w.quantized(T(p + "self_attn.v_proj.weight"), kvdim, hidden, f"{i}.v_proj", "v")
            w.quantized(T(p + "self_attn.o_proj.weight"), hidden, qdim, f"{i}.o_proj", "o")
            w.f32(T(p + "post_attention_layernorm.weight"), hidden)
            w.quantized(T(p + "mlp.up_proj.weight"), ffn, hidden, f"{i}.up_proj", "up")
            w.quantized(T(p + "mlp.gate_proj.weight"), ffn, hidden, f"{i}.gate_proj", "gate")
            w.quantized(T(p + "mlp.down_proj.weight"), hidden, ffn, f"{i}.down_proj", "down")
        w.f32(T("model.norm.weight"), hidden)

    size = os.path.getsize(args.out)
    if size != expected:
        print(f"error: wrote {size:,} bytes, expected {expected:,}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: {args.out} {size:,} bytes")


if __name__ == "__main__":
    main()
