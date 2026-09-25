# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter_stream.py
## @brief Bounded-memory HuggingFace to NNTrainer FP32 converter for Qwen3 MoE

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_CHUNK_MIB = 64
FP32_BYTES = 4


class SafetensorsSource:
    """Resolve tensors from a local HuggingFace safetensors checkpoint.

    Only one shard is kept open at a time. Tensor payloads are requested with
    get_slice(), so callers can materialize bounded slices instead of loading a
    complete shard or model.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.weight_map = self._load_weight_map()
        self._active_context = None
        self._active_handle = None
        self._active_shard = None

    def _load_weight_map(self):
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as index_file:
                index = json.load(index_file)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid or empty weight_map in {index_path}")
            return weight_map

        model_file = self.model_path / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(
                "Expected model.safetensors or model.safetensors.index.json in "
                f"{self.model_path}"
            )
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            return {key: model_file.name for key in handle.keys()}

    def _open_shard(self, shard_name):
        if shard_name == self._active_shard:
            return
        self.close()

        shard_path = self.model_path / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        self._active_context = safe_open(
            str(shard_path), framework="pt", device="cpu"
        )
        self._active_handle = self._active_context.__enter__()
        self._active_shard = shard_name

    def get_slice(self, key):
        if key not in self.weight_map:
            raise KeyError(f"Missing tensor in HuggingFace checkpoint: {key}")
        self._open_shard(self.weight_map[key])
        return self._active_handle.get_slice(key)

    def close(self):
        if self._active_context is not None:
            self._active_context.__exit__(None, None, None)
        self._active_context = None
        self._active_handle = None
        self._active_shard = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Fp32Writer:
    """Write safetensors tensors in NNTrainer's positional FP32 layout."""

    def __init__(self, source, output, chunk_bytes):
        self.source = source
        self.output = output
        self.chunk_bytes = chunk_bytes
        self.elements_written = 0

    @staticmethod
    def _check_shape(key, actual, expected):
        actual = tuple(actual)
        expected = tuple(expected)
        if actual != expected:
            raise ValueError(
                f"Unexpected shape for {key}: expected {expected}, got {actual}"
            )

    def _write(self, tensor, transpose):
        tensor = tensor.detach().cpu()
        if transpose:
            tensor = tensor.transpose(0, 1).contiguous()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        array = tensor.numpy()
        array.tofile(self.output)
        self.elements_written += array.size

    def write_tensor(self, key, expected_shape, transpose=False):
        tensor_slice = self.source.get_slice(key)
        shape = tuple(tensor_slice.get_shape())
        self._check_shape(key, shape, expected_shape)

        if transpose:
            if len(shape) != 2:
                raise ValueError(f"Cannot transpose non-matrix tensor: {key}")
            source_rows, source_columns = shape
            columns_per_chunk = max(
                1, self.chunk_bytes // (source_rows * FP32_BYTES)
            )
            for column in range(0, source_columns, columns_per_chunk):
                end = min(column + columns_per_chunk, source_columns)
                self._write(tensor_slice[:, column:end], transpose=True)
            return

        if len(shape) == 1:
            elements_per_chunk = max(1, self.chunk_bytes // FP32_BYTES)
            for element in range(0, shape[0], elements_per_chunk):
                end = min(element + elements_per_chunk, shape[0])
                self._write(tensor_slice[element:end], transpose=False)
            return

        if len(shape) != 2:
            raise ValueError(f"Expected a vector or matrix tensor: {key}")
        row_bytes = shape[1] * FP32_BYTES
        rows_per_chunk = max(1, self.chunk_bytes // row_bytes)
        for row in range(0, shape[0], rows_per_chunk):
            end = min(row + rows_per_chunk, shape[0])
            self._write(tensor_slice[row:end, :], transpose=False)


def require_positive_int(config, key):
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.json requires a positive integer '{key}'")
    return value


def load_model_plan(model_path):
    config_path = Path(model_path) / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config.json requires architectures[0]")
    if architectures[0] != "Qwen3MoeForCausalLM":
        raise ValueError(
            "Streaming conversion only supports Qwen3MoeForCausalLM, got "
            f"{architectures[0]}"
        )

    hidden_size = require_positive_int(config, "hidden_size")
    num_attention_heads = require_positive_int(config, "num_attention_heads")
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError("config.json requires a positive integer 'head_dim'")

    return {
        "hidden_size": hidden_size,
        "vocab_size": require_positive_int(config, "vocab_size"),
        "num_layers": require_positive_int(config, "num_hidden_layers"),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": require_positive_int(
            config, "num_key_value_heads"
        ),
        "head_dim": head_dim,
        "moe_intermediate_size": require_positive_int(
            config, "moe_intermediate_size"
        ),
        "num_experts": require_positive_int(config, "num_experts"),
        "tied_embeddings": bool(config.get("tie_word_embeddings", False)),
    }


def write_qwen3_moe(source, output, model, chunk_bytes):
    writer = Fp32Writer(source, output, chunk_bytes)
    hidden_size = model["hidden_size"]
    query_width = model["num_attention_heads"] * model["head_dim"]
    kv_width = model["num_key_value_heads"] * model["head_dim"]
    moe_width = model["moe_intermediate_size"]

    writer.write_tensor(
        "model.embed_tokens.weight",
        (model["vocab_size"], hidden_size),
    )

    for layer in range(model["num_layers"]):
        prefix = f"model.layers.{layer}."
        writer.write_tensor(
            f"{prefix}input_layernorm.weight", (hidden_size,)
        )
        writer.write_tensor(
            f"{prefix}self_attn.q_proj.weight",
            (query_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}self_attn.q_norm.weight", (model["head_dim"],)
        )
        writer.write_tensor(
            f"{prefix}self_attn.k_proj.weight",
            (kv_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}self_attn.k_norm.weight", (model["head_dim"],)
        )
        writer.write_tensor(
            f"{prefix}self_attn.v_proj.weight",
            (kv_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}self_attn.o_proj.weight",
            (hidden_size, query_width),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}post_attention_layernorm.weight", (hidden_size,)
        )
        writer.write_tensor(
            f"{prefix}mlp.gate.weight",
            (model["num_experts"], hidden_size),
            transpose=True,
        )

        for expert in range(model["num_experts"]):
            expert_prefix = f"{prefix}mlp.experts.{expert}."
            for projection, shape in (
                ("up_proj", (moe_width, hidden_size)),
                ("gate_proj", (moe_width, hidden_size)),
                ("down_proj", (hidden_size, moe_width)),
            ):
                writer.write_tensor(
                    f"{expert_prefix}{projection}.weight",
                    shape,
                    transpose=True,
                )

        print(f"  Converted layer {layer + 1}/{model['num_layers']}")

    writer.write_tensor("model.norm.weight", (hidden_size,))
    if not model["tied_embeddings"]:
        writer.write_tensor(
            "lm_head.weight",
            (model["vocab_size"], hidden_size),
            transpose=True,
        )
    return writer.elements_written * FP32_BYTES


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert local HuggingFace Qwen3-MoE safetensors to an NNTrainer "
            "FP32 .bin without loading the full model"
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument(
        "--chunk_size_mib",
        type=int,
        default=DEFAULT_CHUNK_MIB,
        help=f"Maximum FP32 output chunk size (default: {DEFAULT_CHUNK_MIB})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_path = Path(args.output_name)
    partial_path = output_path.with_name(output_path.name + ".part")

    if args.chunk_size_mib <= 0:
        raise ValueError("--chunk_size_mib must be greater than zero")
    if output_path.suffix != ".bin":
        raise ValueError("--output_name must use the .bin extension")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it"
        )

    model = load_model_plan(model_path)
    chunk_bytes = args.chunk_size_mib * 1024 * 1024
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("NNTrainer Qwen3 MoE streaming FP32 converter")
    print(f"  Source: {model_path}")
    print(f"  Target: {output_path}")
    print(f"  Layers: {model['num_layers']}")
    print(f"  Experts per layer: {model['num_experts']}")
    print(f"  Chunk size: {args.chunk_size_mib} MiB")

    try:
        with SafetensorsSource(model_path) as source:
            with partial_path.open("wb") as output:
                expected_size = write_qwen3_moe(
                    source, output, model, chunk_bytes
                )
        actual_size = partial_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                "Unexpected output size: "
                f"expected {expected_size} bytes, got {actual_size}"
            )
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    print(f"Completed: {output_path} ({expected_size} bytes)")


if __name__ == "__main__":
    main()
