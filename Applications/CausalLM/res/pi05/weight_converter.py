# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# Convert lerobot/pi05_base (PyTorch, F32) to nntrainer bins for the pi0.5
# port. The checkpoint has three groups:
#   paligemma_with_expert.paligemma.model.*   backbone = PaliGemma
#                                             (SigLIP vision + Gemma1 18L MQA,
#                                             same as the paligemma port)
#   paligemma_with_expert.gemma_expert.model.*
#                                             action expert = Gemma 1024w 18L
#                                             with adaRMS norms (dense 3x
#                                             width + bias, NOT baked)
#   action_in_proj / action_out_proj /        flow-matching heads
#   time_mlp_in / time_mlp_out
#
# paligemma_with_expert.paligemma.lm_head.weight is a tied duplicate of
# embed_tokens and the expert lm_head is dead weight (never called by the
# flow-matching path) — both are dropped.
#
# Outputs (defaults, all FP32):
#   nntr_pi05_backbone_fp32.bin   bypass layout: decoder layers in graph
#                                 order, final norm (+1.0 baked RMS), no head
#   nntr_pi05_emb_fp32.bin        raw embedding table (lookupEmbedding /
#                                 multimodal injection)
#   nntr_pi05_vision_fp32.bin     vision converter order: conv w+b, pos emb,
#                                 per-layer (ln1, q/k/v/out w+b, ln2,
#                                 fc1/fc2 w+b), post_layernorm, projector
#   nntr_pi05_expert_fp32.bin     per layer (adaRMS w+b, q/k/v/o, adaRMS w+b,
#                                 gate/up/down), final adaRMS, then
#                                 action_in_proj w+b, action_out_proj w+b,
#                                 time_mlp_in w+b, time_mlp_out w+b
#
# NOTE: the expert bin order must match the C++ expert graph's load walk;
# keep the two in sync when the expert model lands.

import argparse
import os

import numpy as np
import torch
from safetensors.torch import load_file

PI = "paligemma_with_expert.paligemma.model."
EXPERT = "paligemma_with_expert.gemma_expert.model."


def fc(t):
    """PyTorch Linear [out, in] -> nntrainer FC record [in, out]. 1-D
    tensors (LayerNorm/gamma-scale vectors and biases) pass through
    untouched — transposing them is meaningless and crashes."""
    if t.dim() == 1:
        return t.float().contiguous()
    return t.float().permute(1, 0).contiguous()


def save_f32(t, f):
    np.asarray(t.contiguous().float(), dtype="<f4").tofile(f)


def save_rms_baked(t, f):
    """Gemma RMSNorm multiplies (1 + gamma): bake the +1.0."""
    save_f32(t.float() + 1.0, f)


def save_wb(params, key, f, weight_transposed=False):
    """FC weight (transposed) followed by bias, for one proj/Linear."""
    save_f32(params[key + ".weight"] if weight_transposed else fc(params[key + ".weight"]), f)
    if key + ".bias" in params:
        save_f32(params[key + ".bias"], f)


def save_ada_rms(params, prefix, f):
    """adaRMS condition projection: dense [3w, w] transposed + bias, raw
    (the +1.0 is applied at runtime before the scale/shift/gate split)."""
    save_wb(params, prefix + "dense", f)


# pi0.5 checkpoints save NO embed_tokens: the tied table lives at
# paligemma.lm_head.weight [257152, 2048] (openpi vocab = 257152).
EMBED_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


def convert_backbone(params, out_path, n_layers, dtype="<f4", weight_transposed=False):
    project = (lambda t: t) if weight_transposed else fc
    def write(t, f):
        np.asarray(t.contiguous().float(), dtype=dtype).tofile(f)
    with open(out_path, "wb") as f:
        # Prefix-only bypass layout: decoder layers and final norm, with no
        # vocabulary head. The raw embedding table is exported separately.
        # 1. decoder layers, graph order
        for i in range(n_layers):
            p = PI + f"language_model.layers.{i}."
            write(params[p + "input_layernorm.weight"].float() + 1.0, f)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                write(project(params[p + f"self_attn.{name}.weight"]), f)
            write(params[p + "post_attention_layernorm.weight"].float() + 1.0, f)
            for name in ("gate_proj", "up_proj", "down_proj"):
                write(project(params[p + f"mlp.{name}.weight"]), f)
            print(f"  backbone layer {i + 1}/{n_layers}")
        # 2. final norm
        write(params[PI + "language_model.norm.weight"].float() + 1.0, f)
    print(f"backbone -> {out_path}")


def save_embedding_bin(params, out_path):
    with open(out_path, "wb") as f:
        save_f32(params[EMBED_KEY], f)  # raw [vocab, hidden]
    print(f"embedding -> {out_path}")


def convert_vision(params, out_path, n_layers, weight_transposed=False):
    V = PI + "vision_tower.vision_model."
    with open(out_path, "wb") as f:
        # patch embedding conv — kernel-first RAW [1152,3,14,14], exactly
        # like the PaliGemma vision converter whose layout the C++ conv2d
        # layer + 3B mm e2e verify (an NHWC permute here silently scrambles
        # every image embedding)
        save_f32(params[V + "embeddings.patch_embedding.weight"], f)
        save_f32(params[V + "embeddings.patch_embedding.bias"], f)
        # position embedding — the checkpoint stores it as [256, 1152]
        # already (no leading batch dim in pi05/lerobot checkpoints)
        save_f32(params[V + "embeddings.position_embedding.weight"], f)
        # encoder layers: ln1, q/k/v/out, ln2, fc1/fc2 (all with bias)
        for i in range(n_layers):
            p = V + f"encoder.layers.{i}."
            save_wb(params, p + "layer_norm1", f, weight_transposed)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                save_wb(params, p + f"self_attn.{name}", f, weight_transposed)
            save_wb(params, p + "layer_norm2", f, weight_transposed)
            save_wb(params, p + "mlp.fc1", f, weight_transposed)
            save_wb(params, p + "mlp.fc2", f, weight_transposed)
            print(f"  vision layer {i + 1}/{n_layers}")
        # post_layernorm + multimodal projector (1152 -> 2048, with bias)
        save_wb(params, V + "post_layernorm", f, weight_transposed)
        save_wb(params, PI + "multi_modal_projector.linear", f, weight_transposed)
    print(f"vision -> {out_path}")


def convert_expert(params, out_path, n_layers):
    with open(out_path, "wb") as f:
        for i in range(n_layers):
            p = EXPERT + f"layers.{i}."
            save_ada_rms(params, p + "input_layernorm.", f)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                save_f32(fc(params[p + f"self_attn.{name}.weight"]), f)
            save_ada_rms(params, p + "post_attention_layernorm.", f)
            for name in ("gate_proj", "up_proj", "down_proj"):
                save_f32(fc(params[p + f"mlp.{name}.weight"]), f)
            print(f"  expert layer {i + 1}/{n_layers}")
        # final adaRMS
        save_ada_rms(params, EXPERT + "norm.", f)
        # flow-matching heads
        save_wb(params, "action_in_proj", f)
        save_wb(params, "action_out_proj", f)
        save_wb(params, "time_mlp_in", f)
        save_wb(params, "time_mlp_out", f)
    print(f"expert -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="dir with model.safetensors (lerobot/pi05_base)")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--num_hidden_layers", type=int, default=18)
    parser.add_argument("--vision_layers", type=int, default=27)
    parser.add_argument("--backbone_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--weight-transposed", action="store_true",
                        help="backbone/vision [output,input]; expert format stays unchanged")
    args = parser.parse_args()

    path = os.path.join(args.model_path, "model.safetensors")
    params = load_file(path)
    print(f"loaded {len(params)} tensors from {path}")

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, "nntr_pi05_")
    convert_backbone(params, prefix + "backbone_" + args.backbone_dtype + ".bin",
                     args.num_hidden_layers, "<f2" if args.backbone_dtype == "fp16" else "<f4",
                     args.weight_transposed)
    save_embedding_bin(params, prefix + "emb_fp32.bin")
    convert_vision(params, prefix + "vision_fp32.bin", args.vision_layers, args.weight_transposed)
    convert_expert(params, prefix + "expert_fp32.bin", args.num_hidden_layers)


if __name__ == "__main__":
    main()
