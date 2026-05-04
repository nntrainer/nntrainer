#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert LFM2 weights to the NNTrainer CausalLM binary format."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _dtype_from_name(name: str):
    if name == "fp32":
        return np.float32
    raise ValueError(f"Unsupported data type: {name}")


def _write_tensor(file, tensor: torch.Tensor, dtype) -> int:
    array = tensor.detach().cpu().float().numpy().astype(dtype, copy=False)
    array.tofile(file)
    return array.nbytes


def _write_projection(file, params, key: str, dtype) -> int:
    return _write_tensor(file, params[key].permute(1, 0).contiguous(), dtype)


def save_lfm2_for_nntrainer(model, dtype, file) -> int:
    """Save LFM2 weights in the order used by the NNTrainer graph."""

    params = model.state_dict()
    config = model.config
    total_size = 0

    total_size += _write_tensor(file, params["model.embed_tokens.weight"], dtype)

    for layer_idx, layer_type in enumerate(config.layer_types):
        layer_prefix = f"model.layers.{layer_idx}."

        total_size += _write_tensor(
            file, params[f"{layer_prefix}operator_norm.weight"], dtype
        )

        if layer_type == "full_attention":
            attn_prefix = f"{layer_prefix}self_attn."
            total_size += _write_projection(
                file, params, f"{attn_prefix}q_proj.weight", dtype
            )
            total_size += _write_tensor(
                file, params[f"{attn_prefix}q_layernorm.weight"], dtype
            )
            total_size += _write_projection(
                file, params, f"{attn_prefix}k_proj.weight", dtype
            )
            total_size += _write_tensor(
                file, params[f"{attn_prefix}k_layernorm.weight"], dtype
            )
            total_size += _write_projection(
                file, params, f"{attn_prefix}v_proj.weight", dtype
            )
            total_size += _write_projection(
                file, params, f"{attn_prefix}out_proj.weight", dtype
            )
        elif layer_type == "conv":
            conv_prefix = f"{layer_prefix}conv."
            total_size += _write_projection(
                file, params, f"{conv_prefix}in_proj.weight", dtype
            )
            total_size += _write_tensor(
                file,
                params[f"{conv_prefix}conv.weight"].squeeze(1).contiguous(),
                dtype,
            )
            total_size += _write_projection(
                file, params, f"{conv_prefix}out_proj.weight", dtype
            )
        else:
            raise ValueError(f"Unsupported LFM2 layer type: {layer_type}")

        ffn_prefix = f"{layer_prefix}feed_forward."
        total_size += _write_tensor(
            file, params[f"{layer_prefix}ffn_norm.weight"], dtype
        )
        total_size += _write_projection(file, params, f"{ffn_prefix}w1.weight", dtype)
        total_size += _write_projection(file, params, f"{ffn_prefix}w3.weight", dtype)
        total_size += _write_projection(file, params, f"{ffn_prefix}w2.weight", dtype)

    total_size += _write_tensor(file, params["model.embedding_norm.weight"], dtype)

    if not getattr(config, "tie_word_embeddings", True):
        total_size += _write_projection(file, params, "lm_head.weight", dtype)

    return total_size


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an LFM2 Hugging Face checkpoint for NNTrainer CausalLM."
    )
    parser.add_argument(
        "--model_path",
        default=".",
        help="Path to the Hugging Face LFM2 model directory.",
    )
    parser.add_argument(
        "--output_name",
        default="nntr_lfm2_fp32.bin",
        help="Output NNTrainer binary weight file.",
    )
    parser.add_argument(
        "--data_type",
        default="fp32",
        choices=["fp32"],
        help=(
            "Output floating point data type. "
            "LFM2 short convolution is FP32-only for now."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    dtype = _dtype_from_name(args.data_type)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=args.trust_remote_code
    )
    model.eval()

    with open(args.output_name, "wb") as f_model:
        total_size = save_lfm2_for_nntrainer(model, dtype, f_model)

    print(f"Saved {total_size} bytes to {args.output_name}")


if __name__ == "__main__":
    main()
