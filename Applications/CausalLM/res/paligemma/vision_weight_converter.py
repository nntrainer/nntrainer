## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file vision_weight_converter.py
## @brief weight conversion script for the PaliGemma vision half
##        (SiglipVisionModel + multi-modal projector)
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## The save order must match SiglipVisionModel::constructModel()'s graph:
##   patch_embedding.{weight,bias} -> position_embedding ->
##   per layer {layer_norm1.{w,b}, q/k/v/out_proj.{w(T),b},
##   layer_norm2.{w,b}, fc1.{w(T),b}, fc2.{w(T),b}} ->
##   post_layernorm.{w,b} -> projector linear.{w(T),b}
## (LayerNorm/FC-bias convention matches res/vit/weight_converter.py.)
##
## Key layouts (see weight_converter.py): the checkpoint files use
## vision_tower.vision_model.* / multi_modal_projector.*; recent
## transformers expose model.vision_tower.* / model.multi_modal_projector.*.

import argparse

import numpy as np
import torch

VISION_PREFIXES = ["model.vision_tower.", "vision_tower.vision_model."]
PROJECTOR_PREFIXES = ["model.multi_modal_projector.", "multi_modal_projector."]


def save_weight(weight, file, transpose=False):
    array = weight.detach().cpu().float().numpy()
    if transpose:
        array = array.T
    np.array(array, dtype=np.float32).tofile(file)


def resolve_key(params, suffix, prefixes):
    missing = []
    for prefix in prefixes:
        key = prefix + suffix
        if key in params:
            return params[key]
        missing.append(key)
    raise KeyError(f"none of {missing} found in checkpoint")


def save_siglip_vision_for_nntrainer(params, n_layers, file, weight_transposed=False):
    """Save the SigLIP vision tower + projector in nntrainer graph order."""
    def write_weight(weight, file, transpose=False):
        save_weight(weight, file, transpose=transpose and not weight_transposed)

    # 1. Patch embedding conv (kernel-first, no transpose) + bias
    write_weight(
        resolve_key(params, "embeddings.patch_embedding.weight", VISION_PREFIXES),
        file,
    )
    write_weight(
        resolve_key(params, "embeddings.patch_embedding.bias", VISION_PREFIXES),
        file,
    )

    # 2. Position embedding [num_patches, embed_dim]
    write_weight(
        resolve_key(params, "embeddings.position_embedding.weight", VISION_PREFIXES),
        file,
    )

    # 3. Encoder layers, in graph order
    for i in range(n_layers):
        suffix = f"encoder.layers.{i}."
        write_weight(resolve_key(params, suffix + "layer_norm1.weight", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "layer_norm1.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "self_attn.q_proj.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "self_attn.q_proj.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "self_attn.k_proj.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "self_attn.k_proj.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "self_attn.v_proj.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "self_attn.v_proj.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "self_attn.out_proj.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "self_attn.out_proj.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "layer_norm2.weight", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "layer_norm2.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "mlp.fc1.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "mlp.fc1.bias", VISION_PREFIXES), file)
        write_weight(resolve_key(params, suffix + "mlp.fc2.weight", VISION_PREFIXES), file, transpose=True)
        write_weight(resolve_key(params, suffix + "mlp.fc2.bias", VISION_PREFIXES), file)
        print(f"  vision layer {i + 1}/{n_layers} done")

    # 4. Final post_layernorm
    write_weight(resolve_key(params, "post_layernorm.weight", VISION_PREFIXES), file)
    write_weight(resolve_key(params, "post_layernorm.bias", VISION_PREFIXES), file)

    # 5. Multi-modal projector Linear(vision_hidden -> projection_dim, bias)
    write_weight(resolve_key(params, "linear.weight", PROJECTOR_PREFIXES), file, transpose=True)
    write_weight(resolve_key(params, "linear.bias", PROJECTOR_PREFIXES), file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./paligemma-3b-pt-224")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_paligemma_vision_fp32.bin"
    )
    parser.add_argument("--weight-transposed", action="store_true")
    args = parser.parse_args()

    from transformers import AutoConfig, PaliGemmaForConditionalGeneration

    config = AutoConfig.from_pretrained(args.model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32
    )
    model.eval()

    with open(args.output_name, "wb") as f_model:
        save_siglip_vision_for_nntrainer(
            model.state_dict(), config.vision_config.num_hidden_layers, f_model, args.weight_transposed
        )
    print(f"Saved vision tower + projector to {args.output_name}")
