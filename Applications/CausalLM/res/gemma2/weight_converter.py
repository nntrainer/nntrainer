#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
## @file   weight_converter.py
## @brief  HF Gemma2-2B -> nntrainer layer-graph .bin converter (int4-quantize source).
## @author Jijoong Moon <jijoong.moon@samsung.com>
##
## numpy-only converter: HF Gemma2-2B (model.safetensors, bf16) -> nntrainer
## layer-graph FP32 positional .bin, matching the weight order produced by the
## Gemma2CausalLM graph (= gemma3 weight_converter.py order minus q/k-norm).
##
## This is STEP 1 of the QS4CX-FP16 pipeline (see build_qs4cx.sh): it produces
## the FP32-weight / FP16-norm source .bin that nntr_quantize then quantizes to
## the 1K-benchmark recipe (embedding Q6_K, FC QS4CX, lm_head Q6_K, QS4CX-FP16).
## Use --data_type float32 --norm_fp16 for that source.
##
## No torch / transformers / safetensors lib needed: safetensors is a JSON header
## followed by a raw little-endian data block, and bf16 -> fp32 is an exact widen
## (bf16 = high 16 bits of fp32).
##
## Graph order (whole model):
##   embed_tokens            (as-is, [vocab, hidden], no transpose, no +1)
##   for each layer:
##     input_layernorm       (+1, 1D)
##     q_proj  -> transpose [in, out]
##     k_proj  -> transpose
##     v_proj  -> transpose
##     o_proj  -> transpose
##     post_attention_layernorm   (+1)
##     pre_feedforward_layernorm  (+1)
##     gate_proj -> transpose
##     up_proj   -> transpose
##     down_proj -> transpose
##     post_feedforward_layernorm (+1)
##   model.norm.weight       (+1)
##   (lm_head tied -> not written)
import argparse
import json
import struct

import numpy as np


def load_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_offset = 8 + n
    return hdr, data_offset


def make_reader(path, hdr, data_offset):
    DTYPE = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # widened to fp32 below
        "I8": (np.int8, 1),
    }

    def read(name):
        meta = hdr[name]
        dt = meta["dtype"]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        np_dt, _ = DTYPE[dt]
        with open(path, "rb") as f:
            f.seek(data_offset + beg)
            raw = f.read(end - beg)
        arr = np.frombuffer(raw, dtype=np_dt)
        if dt == "BF16":
            # exact widen: bf16 bits are the top 16 bits of fp32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr.reshape(shape)

    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="HF model dir containing model.safetensors + config.json")
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--data_type", choices=["float32", "float16"],
                    default="float32")
    ap.add_argument("--norm_fp16", action="store_true",
                    help="write RMSNorm gammas as FP16 (mixed bin for the "
                         "FP32-weight / FP16-activation quantize path)")
    args = ap.parse_args()

    out_dtype = np.float32 if args.data_type == "float32" else np.float16
    norm_fp16 = args.norm_fp16

    cfg = json.load(open(f"{args.model_path}/config.json"))
    n_layers = cfg["num_hidden_layers"]

    st = f"{args.model_path}/model.safetensors"
    hdr, off = load_safetensors_header(st)
    read = make_reader(st, hdr, off)

    total = [0]
    f = open(args.output_name, "wb")

    def w(arr, is_rms=False, transpose=False):
        a = arr
        if is_rms:
            a = a + 1.0
        if transpose:
            a = a.T
        # RMSNorm gammas are written as FP16 even when the rest of the bin is
        # FP32: the layer-graph runs with FP16 activations (model_tensor_type
        # *-FP16), and packed=false norm layers take the activation dtype, so
        # the model loads gamma as FP16. The FC/embed weights stay FP32 so
        # nntr_quantize can quantize them (quantize requires FP32 source).
        dt = np.float16 if (is_rms and norm_fp16) else out_dtype
        a = np.ascontiguousarray(a, dtype=dt)
        a.tofile(f)
        total[0] += a.nbytes

    def proj(layer, name):
        return read(f"model.layers.{layer}.{name}.weight")

    # embedding (as-is)
    w(read("model.embed_tokens.weight"))

    for i in range(n_layers):
        lp = f"model.layers.{i}."
        # attention
        w(read(lp + "input_layernorm.weight"), is_rms=True)
        w(proj(i, "self_attn.q_proj"), transpose=True)
        w(proj(i, "self_attn.k_proj"), transpose=True)
        w(proj(i, "self_attn.v_proj"), transpose=True)
        w(proj(i, "self_attn.o_proj"), transpose=True)
        # feed forward
        w(read(lp + "post_attention_layernorm.weight"), is_rms=True)
        w(read(lp + "pre_feedforward_layernorm.weight"), is_rms=True)
        w(proj(i, "mlp.gate_proj"), transpose=True)
        w(proj(i, "mlp.up_proj"), transpose=True)
        w(proj(i, "mlp.down_proj"), transpose=True)
        w(read(lp + "post_feedforward_layernorm.weight"), is_rms=True)

    # final norm
    w(read("model.norm.weight"), is_rms=True)

    f.close()
    print(f"wrote {args.output_name}: {total[0]} bytes "
          f"({total[0]/1e9:.2f} GB), {n_layers} layers, dtype={args.data_type}")


if __name__ == "__main__":
    main()
