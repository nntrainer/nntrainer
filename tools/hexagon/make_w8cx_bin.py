#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    make_w8cx_bin.py
# @brief   Build the Qwen3 W8_CX checkpoint (.bin) from a HuggingFace directory
# @author  dlwlzzero <dlwlzzero@gmail.com>
"""Build the Qwen3 W8_CX checkpoint (.bin) straight from a HuggingFace directory.

`Qwen3W8cxBin` (Applications/CausalLM/hexagon/qwen3_w8cx_bin.cpp) reads a
header-less file: every 2-D weight as int8 [N][K] followed by N fp32 scales,
every norm as fp32, in the order

    embed_tokens
    per layer: input_layernorm, q_proj, q_norm, k_proj, k_norm, v_proj,
               o_proj, post_attention_layernorm, up_proj, gate_proj, down_proj
    final norm

The quantizer is the symmetric per-output-channel W8_CX primitive
(`__fallback_quant_nxk_w8cx_f32`): scale = absmax / 127 (stored as the
dequant multiplier), code = lround(x * 127 / absmax) clamped to [-127, 127],
all in float32. HF linear weights are already [out][in] = [N][K], so no
transpose is needed. For Qwen3-0.6B the output is 598,230,528 bytes.

The nntrainer `nntr_quantize --fc_dtype W8_CX` path lives on the hvx_m3
branch only; this script needs numpy and the HF safetensors file, nothing
else, and produces the same bytes (same primitive, same order).

Usage:
    make_w8cx_bin.py <hf_dir> <out.bin> [--layers N]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


def load_safetensors(path):
    """Return {name: (dtype, shape, memoryview)} for a .safetensors file."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + hlen)
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta["dtype"], tuple(meta["shape"]), data[a:b])
    return out


def to_f32(entry):
    dtype, shape, raw = entry
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise ValueError(f"unsupported dtype {dtype}")


def lround_half_away(y):
    """std::lround: round half away from zero (numpy rounds half to even)."""
    return (np.sign(y) * np.floor(np.abs(y) + np.float32(0.5))).astype(np.int32)


def quant_w8cx(w):
    """w: float32 [N][K] -> (int8 [N][K], float32 [N]) exactly like the C primitive."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1).astype(np.float32)               # float32 max
    scale = (amax / np.float32(127.0)).astype(np.float32)          # float32 division
    inv = np.where(amax > 0, np.float32(127.0) / amax, np.float32(0.0)).astype(np.float32)
    y = (w * inv[:, None]).astype(np.float32)                      # float32 multiply
    q = np.clip(lround_half_away(y), -127, 127).astype(np.int8)
    return q, scale


class Writer:
    def __init__(self, fh):
        self.fh = fh
        self.nbytes = 0

    def f32(self, a, n):
        a = np.ascontiguousarray(a, dtype=np.float32)
        assert a.size == n, (a.shape, n)
        self.fh.write(a.tobytes())
        self.nbytes += a.nbytes

    def quantized(self, w, n, k, label):
        assert w.shape == (n, k), (label, w.shape, (n, k))
        q, s = quant_w8cx(w)
        self.fh.write(q.tobytes())
        self.fh.write(s.tobytes())
        self.nbytes += q.nbytes + s.nbytes
        print(f"  {label:44s} [{n}x{k}] int8 + {n} scales", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir", help="HuggingFace model directory (config.json + model.safetensors)")
    ap.add_argument("out", help="output .bin path")
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (bring-up images)")
    args = ap.parse_args()

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

    expected = (qb(vocab, hidden)
                + n_layers * (4 * hidden + qb(qdim, hidden) + 4 * head_dim + qb(kvdim, hidden)
                              + 4 * head_dim + qb(kvdim, hidden) + qb(hidden, qdim) + 4 * hidden
                              + qb(ffn, hidden) + qb(ffn, hidden) + qb(hidden, ffn))
                + 4 * hidden)

    print(f"hf={args.hf_dir} layers={n_layers} hidden={hidden} ffn={ffn} vocab={vocab} "
          f"heads={n_heads}/{n_kv} head_dim={head_dim} -> {args.out} ({expected:,} bytes)")

    with open(args.out, "wb") as fh:
        w = Writer(fh)
        w.quantized(T("model.embed_tokens.weight"), vocab, hidden, "embed_tokens")
        for i in range(n_layers):
            p = f"model.layers.{i}."
            w.f32(T(p + "input_layernorm.weight"), hidden)
            w.quantized(T(p + "self_attn.q_proj.weight"), qdim, hidden, f"{i}.q_proj")
            w.f32(T(p + "self_attn.q_norm.weight"), head_dim)
            w.quantized(T(p + "self_attn.k_proj.weight"), kvdim, hidden, f"{i}.k_proj")
            w.f32(T(p + "self_attn.k_norm.weight"), head_dim)
            w.quantized(T(p + "self_attn.v_proj.weight"), kvdim, hidden, f"{i}.v_proj")
            w.quantized(T(p + "self_attn.o_proj.weight"), hidden, qdim, f"{i}.o_proj")
            w.f32(T(p + "post_attention_layernorm.weight"), hidden)
            w.quantized(T(p + "mlp.up_proj.weight"), ffn, hidden, f"{i}.up_proj")
            w.quantized(T(p + "mlp.gate_proj.weight"), ffn, hidden, f"{i}.gate_proj")
            w.quantized(T(p + "mlp.down_proj.weight"), hidden, ffn, f"{i}.down_proj")
        w.f32(T("model.norm.weight"), hidden)

    size = os.path.getsize(args.out)
    if size != expected:
        print(f"error: wrote {size:,} bytes, expected {expected:,}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: {args.out} {size:,} bytes")


if __name__ == "__main__":
    main()
