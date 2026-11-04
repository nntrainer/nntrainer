# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

# @file weight_converter.py
# @brief weight conversion script for kalm-embedding model
# @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}


def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert torch tensor to contiguous numpy array."""
    if transpose:
        tensor = tensor.permute(1, 0)

    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"

    if output_name.endswith(".safetensors"):
        return output_name

    return output_name + ".safetensors"


def resolve_prefix(params):
    """Resolve the Qwen2 backbone key prefix inside a (possibly wrapped) state dict.

    Plain ``AutoModel`` (Qwen2Model) yields keys like ``embed_tokens.weight``;
    a ``SentenceTransformer`` wrapper yields ``0.auto_model.embed_tokens.weight``.
    """
    candidates = ["", "model.", "0.auto_model.", "auto_model."]
    for prefix in candidates:
        if f"{prefix}embed_tokens.weight" in params:
            return prefix

    raise KeyError("Cannot find Qwen2 embedding backbone weights in state_dict")


def save_kalm_embedding_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save weights as nntrainer binary format for KaLM embedding."""

    prefix = resolve_prefix(params)

    def save_weight(tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        arr.tofile(file)

    def save_projection(layer_name, proj_name):
        """Save projection weight, handling optional LoRA weights."""
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
        """Save attention layer weights (Qwen2 backbone uses bias on Q/K/V)."""
        save_weight(params[f"{layer_name}input_layernorm.weight"])

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")

            bias_key = f"{layer_name}self_attn.{proj}.bias"
            if bias_key in params:
                save_weight(params[bias_key])

    def save_feed_forward(layer_name):
        """Save feed-forward layer weights in nntrainer up, gate, down order."""
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])

        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    save_weight(params[f"{prefix}embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"{prefix}layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params[f"{prefix}norm.weight"])


def collect_kalm_embedding_for_nntrainer(params, n_layers, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs."""
    prefix = resolve_prefix(params)
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    def add_projection(nntr_name, layer_name, proj_name):
        base_key = f"{layer_name}{proj_name}.base_layer.weight"
        weight_key = f"{layer_name}{proj_name}.weight"
        lora_a_key = f"{layer_name}{proj_name}.lora_A.default.weight"

        if lora_a_key in params:
            add(nntr_name, params[base_key], transpose=True)
            return

        add(nntr_name, params[weight_key], transpose=True)

    add("embedding0:Embedding", params[f"{prefix}embed_tokens.weight"])

    for layer_idx in range(n_layers):
        hf_prefix = f"{prefix}layers.{layer_idx}."
        nntr_prefix = f"layer{layer_idx}"

        add(
            f"{nntr_prefix}_attention_norm:gamma",
            params[f"{hf_prefix}input_layernorm.weight"],
        )

        for proj, nntr in [("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv")]:
            add_projection(
                f"{nntr_prefix}_{nntr}:weight",
                hf_prefix,
                f"self_attn.{proj}",
            )
            bias_key = f"{hf_prefix}self_attn.{proj}.bias"
            if bias_key in params:
                add(f"{nntr_prefix}_{nntr}:bias", params[bias_key])

        add_projection(
            f"{nntr_prefix}_attention_out:weight",
            hf_prefix,
            "self_attn.o_proj",
        )

        add(
            f"{nntr_prefix}_ffn_norm:gamma",
            params[f"{hf_prefix}post_attention_layernorm.weight"],
        )

        add_projection(f"{nntr_prefix}_ffn_up:weight", hf_prefix, "mlp.up_proj")
        add_projection(f"{nntr_prefix}_ffn_gate:weight", hf_prefix, "mlp.gate_proj")
        add_projection(f"{nntr_prefix}_ffn_down:weight", hf_prefix, "mlp.down_proj")

    add("output_norm:gamma", params[f"{prefix}norm.weight"])

    return weights


def save_safetensors(weights, output_path, dtype):
    """Write weights to a safetensors file."""
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
        default="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        help="Hugging Face model path or local model directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_kalm_embedding_fp32.bin",
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
    """Convert KaLM embedding Hugging Face weights to nntrainer weight format."""
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    params = model.state_dict()

    if args.safetensors:
        output_name = get_safetensors_output_name(args.output_name)

        weights = collect_kalm_embedding_for_nntrainer(
            params,
            config.num_hidden_layers,
            args.data_type,
        )

        save_safetensors(weights, output_name, args.data_type)
        return

    with open(args.output_name, "wb") as output_file:
        save_kalm_embedding_for_nntrainer(
            params,
            config.num_hidden_layers,
            args.data_type,
            output_file,
        )

    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()
