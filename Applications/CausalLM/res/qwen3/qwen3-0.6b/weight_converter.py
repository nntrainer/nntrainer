# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief weight conversion script for qwen3 model
## @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}


def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert torch tensor to contiguous numpy array."""
    if transpose:
        tensor = tensor.permute(1, 0)

    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_tie_word_embeddings(config):
    """Return whether the model uses tied word embeddings."""
    return getattr(config, "tie_word_embeddings", True)


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"

    if output_name.endswith(".safetensors"):
        return output_name

    return output_name + ".safetensors"


def save_qwen3_for_nntrainer(
    params,
    n_layers,
    dtype,
    file,
    tie_word_embeddings=True,
):
    """Convert and save weights as nntrainer binary format for Qwen3 model."""

    def save_weight(tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        arr.tofile(file)

    def save_projection(layer_name, proj_name):
        """Save projection weight.

        If LoRA weights exist, save base, LoRA A, and LoRA B weights in order.
        """
        base_key = f"{layer_name}{proj_name}.base_layer.weight"
        weight_key = f"{layer_name}{proj_name}.weight"
        lora_a_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        lora_b_key = f"{layer_name}{proj_name}.lora_B.default.weight"

        if lora_a_key in params:
            save_weight(params[base_key], transpose=True)
            save_weight(params[lora_a_key], transpose=True)
            save_weight(params[lora_b_key], transpose=True)
            return

        save_weight(params[weight_key], transpose=True)

    def save_attention(layer_name):
        """Save attention layer weights."""
        save_weight(params[f"{layer_name}input_layernorm.weight"])

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")

            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(params[proj_norm_name])

    def save_feed_forward(layer_name):
        """Save feed-forward layer weights."""
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])

        # Keep the existing nntrainer binary weight order.
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    save_weight(params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["model.norm.weight"])

    # For tied embedding models, lm_head shares model.embed_tokens.weight.
    # nntrainer tie_word_embeddings lm_head path does not read a separate weight.
    if not tie_word_embeddings:
        save_weight(params["lm_head.weight"], transpose=True)


def collect_qwen3_for_nntrainer(
    params,
    n_layers,
    dtype,
    tie_word_embeddings=True,
):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs."""
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    def add_projection(nntr_name, layer_name, proj_name):
        """Add projection weight for safetensors export.

        Note: this safetensors path exports the main projection weight only.
        If LoRA safetensors export is needed later, LoRA tensor names should be
        defined and added here explicitly.
        """
        base_key = f"{layer_name}{proj_name}.base_layer.weight"
        weight_key = f"{layer_name}{proj_name}.weight"
        lora_a_key = f"{layer_name}{proj_name}.lora_A.default.weight"

        if lora_a_key in params:
            add(nntr_name, params[base_key], transpose=True)
            return

        add(nntr_name, params[weight_key], transpose=True)

    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        hf_prefix = f"model.layers.{layer_idx}."
        nntr_prefix = f"layer{layer_idx}"

        add(
            f"{nntr_prefix}_attention_norm:gamma",
            params[f"{hf_prefix}input_layernorm.weight"],
        )

        add_projection(
            f"{nntr_prefix}_wq:weight",
            hf_prefix,
            "self_attn.q_proj",
        )

        q_norm_key = f"{hf_prefix}self_attn.q_norm.weight"
        if q_norm_key in params:
            add(f"{nntr_prefix}_q_norm:gamma", params[q_norm_key])

        add_projection(
            f"{nntr_prefix}_wk:weight",
            hf_prefix,
            "self_attn.k_proj",
        )

        k_norm_key = f"{hf_prefix}self_attn.k_norm.weight"
        if k_norm_key in params:
            add(f"{nntr_prefix}_k_norm:gamma", params[k_norm_key])

        add_projection(
            f"{nntr_prefix}_wv:weight",
            hf_prefix,
            "self_attn.v_proj",
        )

        add_projection(
            f"{nntr_prefix}_attention_out:weight",
            hf_prefix,
            "self_attn.o_proj",
        )

        add(
            f"{nntr_prefix}_ffn_norm:gamma",
            params[f"{hf_prefix}post_attention_layernorm.weight"],
        )

        # Match the existing nntrainer Qwen3 layer naming.
        add_projection(
            f"{nntr_prefix}_ffn_gate:weight",
            hf_prefix,
            "mlp.gate_proj",
        )

        add_projection(
            f"{nntr_prefix}_ffn_up:weight",
            hf_prefix,
            "mlp.up_proj",
        )

        add_projection(
            f"{nntr_prefix}_ffn_down:weight",
            hf_prefix,
            "mlp.down_proj",
        )

    add("output_norm:gamma", params["model.norm.weight"])

    if not tie_word_embeddings:
        add(
            "output_of_causallm:weight",
            params["lm_head.weight"],
            transpose=True,
        )

    return weights


def save_safetensors(weights, output_path, dtype):
    """Write weights to a safetensors file.

    This writes the safetensors format directly to avoid an extra dependency.
    """
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }

        raw_buffers.append(arr.tobytes(order="C"))
        offset += nbytes

    header = {"__metadata__": metadata}
    header.update(tensor_meta)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)

        for buffer in raw_buffers:
            output_file.write(buffer)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {offset / 1e9:.2f} GB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model path or local model directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_qwen3_0.6b_fp32.bin",
        help="Output weight file path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32"],
        help="Output data type",
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )

    return parser.parse_args()


def main():
    """Convert Qwen3 Hugging Face weights to nntrainer weight format."""
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    tie_word_embeddings = get_tie_word_embeddings(config)
    print(f"tie_word_embeddings: {tie_word_embeddings}")

    params = model.state_dict()

    if args.safetensors:
        output_name = get_safetensors_output_name(args.output_name)

        weights = collect_qwen3_for_nntrainer(
            params,
            config.num_hidden_layers,
            args.data_type,
            tie_word_embeddings=tie_word_embeddings,
        )

        save_safetensors(weights, output_name, args.data_type)
        return

    with open(args.output_name, "wb") as output_file:
        save_qwen3_for_nntrainer(
            params,
            config.num_hidden_layers,
            args.data_type,
            output_file,
            tie_word_embeddings=tie_word_embeddings,
        )

    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()
