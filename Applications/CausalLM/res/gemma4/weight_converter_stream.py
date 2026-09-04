# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter_stream.py
## @brief Bounded-memory HuggingFace to NNTrainer FP32 converter for Gemma4

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_CHUNK_MIB = 64
FP32_BYTES = 4
MODEL_PREFIXES = (
    "model.language_model.",
    "language_model.",
    "model.",
    "",
)


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

    def resolve_key(self, relative_key):
        for prefix in MODEL_PREFIXES:
            key = prefix + relative_key
            if key in self.weight_map:
                return key
        raise KeyError(
            f"Could not find '{relative_key}' under prefixes {MODEL_PREFIXES}"
        )

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

    def get_slice(self, relative_key):
        key = self.resolve_key(relative_key)
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

    @staticmethod
    def _compose_slice(parent, child, dimension):
        selected = range(dimension)[parent][child]
        if isinstance(selected, int):
            return selected
        return slice(selected.start, selected.stop, selected.step)

    @classmethod
    def _compose_index(cls, view_index, child_index, source_shape):
        source_index = []
        logical_axis = 0
        for axis, parent in enumerate(view_index):
            if isinstance(parent, int):
                source_index.append(parent)
                continue
            source_index.append(
                cls._compose_slice(
                    parent, child_index[logical_axis], source_shape[axis]
                )
            )
            logical_axis += 1
        return tuple(source_index)

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

    def _write_view(self, tensor_slice, source_shape, view_index, shape,
                    transpose):
        if len(shape) == 0:
            self._write(tensor_slice[view_index], transpose=False)
            return

        if len(shape) == 1:
            elements_per_chunk = max(1, self.chunk_bytes // FP32_BYTES)
            ranges = (
                (slice(start, min(start + elements_per_chunk, shape[0])),)
                for start in range(0, shape[0], elements_per_chunk)
            )
        elif len(shape) == 2 and transpose:
            source_rows, source_columns = shape
            columns_per_chunk = max(
                1, self.chunk_bytes // max(1, source_rows * FP32_BYTES)
            )
            ranges = (
                (slice(None), slice(start, min(start + columns_per_chunk,
                                               source_columns)))
                for start in range(0, source_columns, columns_per_chunk)
            )
        elif len(shape) == 2:
            source_rows, source_columns = shape
            rows_per_chunk = max(
                1, self.chunk_bytes // max(1, source_columns * FP32_BYTES)
            )
            ranges = (
                (slice(start, min(start + rows_per_chunk, source_rows)),
                 slice(None))
                for start in range(0, source_rows, rows_per_chunk)
            )
        else:
            raise ValueError(f"Expected a scalar, vector, or matrix, got {shape}")

        for child_index in ranges:
            source_index = self._compose_index(
                view_index, child_index, source_shape
            )
            self._write(tensor_slice[source_index], transpose)

    def write_tensor(self, key, expected_shape, transpose=False):
        tensor_slice = self.source.get_slice(key)
        source_shape = tuple(tensor_slice.get_shape())
        self._check_shape(key, source_shape, expected_shape)
        view_index = tuple(slice(None) for _ in source_shape)
        self._write_view(
            tensor_slice, source_shape, view_index, source_shape, transpose
        )

    def write_view(self, key, source_shape, view_index, expected_shape,
                   transpose=False):
        tensor_slice = self.source.get_slice(key)
        actual_shape = tuple(tensor_slice.get_shape())
        self._check_shape(key, actual_shape, source_shape)
        if len(view_index) != len(source_shape):
            raise ValueError(f"Invalid view rank for {key}: {view_index}")

        view_shape = []
        for dimension, index in zip(source_shape, view_index):
            if isinstance(index, int):
                range(dimension)[index]
            elif isinstance(index, slice):
                view_shape.append(len(range(dimension)[index]))
            else:
                raise ValueError(f"Invalid view index for {key}: {index}")
        self._check_shape(f"{key} view", view_shape, expected_shape)
        self._write_view(
            tensor_slice,
            tuple(source_shape),
            tuple(view_index),
            tuple(expected_shape),
            transpose,
        )


def require_positive_int(config, key):
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.json requires a positive integer '{key}'")
    return value


def optional_positive_int(config, key, default):
    value = config.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.json requires a positive integer '{key}'")
    return value


def optional_nonnegative_int(config, key, default=0):
    value = config.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"config.json requires a non-negative integer '{key}'"
        )
    return value


def load_model_plan(model_path):
    config_path = Path(model_path) / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        root_config = json.load(config_file)

    architectures = root_config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config.json requires architectures[0]")
    supported = ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
    if architectures[0] not in supported:
        raise ValueError(
            "Streaming conversion only supports Gemma4, got "
            f"{architectures[0]}"
        )

    config = root_config.get("text_config", root_config)
    if not isinstance(config, dict):
        raise ValueError("config.json text_config must be an object")

    hidden_size = require_positive_int(config, "hidden_size")
    vocab_size = require_positive_int(config, "vocab_size")
    num_layers = require_positive_int(config, "num_hidden_layers")
    num_attention_heads = require_positive_int(config, "num_attention_heads")
    head_dim = optional_positive_int(
        config, "head_dim", hidden_size // num_attention_heads
    )
    num_key_value_heads = optional_positive_int(
        config, "num_key_value_heads", num_attention_heads
    )
    global_head_dim = optional_positive_int(config, "global_head_dim", head_dim)
    num_global_key_value_heads = optional_positive_int(
        config, "num_global_key_value_heads", num_key_value_heads
    )

    layer_types = config.get(
        "layer_types", ["sliding_attention"] * num_layers
    )
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("layer_types must contain one entry per hidden layer")
    for layer_type in layer_types:
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported Gemma4 layer type: {layer_type}")

    num_kv_shared_layers = optional_nonnegative_int(
        config, "num_kv_shared_layers"
    )
    if num_kv_shared_layers >= num_layers:
        raise ValueError("num_kv_shared_layers must be less than num_hidden_layers")

    ple_size = optional_nonnegative_int(
        config, "hidden_size_per_layer_input"
    )
    ple_vocab_size = optional_positive_int(
        config, "vocab_size_per_layer_input", vocab_size
    )
    enable_moe = bool(config.get("enable_moe_block", False))
    if enable_moe:
        moe_intermediate_size = require_positive_int(
            config, "moe_intermediate_size"
        )
        num_experts = require_positive_int(config, "num_experts")
    else:
        moe_intermediate_size = 0
        num_experts = 0

    tied_embeddings = bool(
        config.get(
            "tie_word_embeddings",
            root_config.get("tie_word_embeddings", True),
        )
    )
    return {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "global_head_dim": global_head_dim,
        "num_global_key_value_heads": num_global_key_value_heads,
        "intermediate_size": require_positive_int(config, "intermediate_size"),
        "moe_intermediate_size": moe_intermediate_size,
        "num_experts": num_experts,
        "num_kv_shared_layers": num_kv_shared_layers,
        "layer_types": layer_types,
        "attention_k_eq_v": bool(config.get("attention_k_eq_v", False)),
        "enable_moe": enable_moe,
        "ple_size": ple_size,
        "ple_vocab_size": ple_vocab_size,
        "tied_embeddings": tied_embeddings,
    }


def write_gemma4(source, output, model, chunk_bytes):
    writer = Fp32Writer(source, output, chunk_bytes)
    hidden_size = model["hidden_size"]
    dense_width = model["intermediate_size"]
    first_kv_shared_layer = (
        model["num_layers"] - model["num_kv_shared_layers"]
    )

    writer.write_tensor(
        "embed_tokens.weight", (model["vocab_size"], hidden_size)
    )

    for layer in range(model["num_layers"]):
        prefix = f"layers.{layer}."
        is_sliding = model["layer_types"][layer] == "sliding_attention"
        is_kv_shared = layer >= first_kv_shared_layer
        head_dim = model["head_dim"] if is_sliding else model["global_head_dim"]
        kv_heads = (
            model["num_key_value_heads"]
            if is_sliding or not model["attention_k_eq_v"]
            else model["num_global_key_value_heads"]
        )
        query_width = model["num_attention_heads"] * head_dim
        kv_width = kv_heads * head_dim

        writer.write_tensor(f"{prefix}input_layernorm.weight", (hidden_size,))
        writer.write_tensor(
            f"{prefix}self_attn.q_proj.weight",
            (query_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(f"{prefix}self_attn.q_norm.weight", (head_dim,))

        if not is_kv_shared:
            writer.write_tensor(
                f"{prefix}self_attn.k_proj.weight",
                (kv_width, hidden_size),
                transpose=True,
            )
            writer.write_tensor(f"{prefix}self_attn.k_norm.weight", (head_dim,))
            if not (model["attention_k_eq_v"] and not is_sliding):
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
            f"{prefix}pre_feedforward_layernorm.weight", (hidden_size,)
        )
        writer.write_tensor(
            f"{prefix}mlp.gate_proj.weight",
            (dense_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}mlp.up_proj.weight",
            (dense_width, hidden_size),
            transpose=True,
        )
        writer.write_tensor(
            f"{prefix}mlp.down_proj.weight",
            (hidden_size, dense_width),
            transpose=True,
        )

        if model["enable_moe"]:
            writer.write_tensor(
                f"{prefix}post_feedforward_layernorm_1.weight", (hidden_size,)
            )
            writer.write_tensor(
                f"{prefix}pre_feedforward_layernorm_2.weight", (hidden_size,)
            )
            writer.write_tensor(
                f"{prefix}router.proj.weight",
                (model["num_experts"], hidden_size),
                transpose=True,
            )
            writer.write_tensor(f"{prefix}router.scale", (hidden_size,))
            writer.write_tensor(
                f"{prefix}router.per_expert_scale", (model["num_experts"],)
            )

            moe_width = model["moe_intermediate_size"]
            gate_up_shape = (model["num_experts"], 2 * moe_width, hidden_size)
            down_shape = (model["num_experts"], hidden_size, moe_width)
            for expert in range(model["num_experts"]):
                writer.write_view(
                    f"{prefix}experts.gate_up_proj",
                    gate_up_shape,
                    (expert, slice(0, moe_width), slice(None)),
                    (moe_width, hidden_size),
                    transpose=True,
                )
                writer.write_view(
                    f"{prefix}experts.gate_up_proj",
                    gate_up_shape,
                    (expert, slice(moe_width, None), slice(None)),
                    (moe_width, hidden_size),
                    transpose=True,
                )
                writer.write_view(
                    f"{prefix}experts.down_proj",
                    down_shape,
                    (expert, slice(None), slice(None)),
                    (hidden_size, moe_width),
                    transpose=True,
                )

            writer.write_tensor(
                f"{prefix}post_feedforward_layernorm_2.weight", (hidden_size,)
            )

        writer.write_tensor(
            f"{prefix}post_feedforward_layernorm.weight", (hidden_size,)
        )

        if model["ple_size"] > 0:
            ple_size = model["ple_size"]
            total_ple_size = model["num_layers"] * ple_size
            writer.write_tensor(
                f"{prefix}per_layer_input_gate.weight",
                (ple_size, hidden_size),
                transpose=True,
            )
            if layer == 0:
                writer.write_tensor(
                    "embed_tokens_per_layer.weight",
                    (model["ple_vocab_size"], total_ple_size),
                )
                writer.write_tensor(
                    "per_layer_model_projection.weight",
                    (total_ple_size, hidden_size),
                    transpose=True,
                )
                writer.write_tensor(
                    "per_layer_projection_norm.weight", (ple_size,)
                )
            writer.write_tensor(
                f"{prefix}per_layer_projection.weight",
                (hidden_size, ple_size),
                transpose=True,
            )
            writer.write_tensor(
                f"{prefix}post_per_layer_input_norm.weight", (hidden_size,)
            )

        writer.write_tensor(f"{prefix}layer_scalar", (1,))
        print(f"  Converted layer {layer + 1}/{model['num_layers']}")

    writer.write_tensor("norm.weight", (hidden_size,))
    if model["tied_embeddings"]:
        # Match the legacy Gemma4 binary converter and NNTrainer lm-head save.
        writer.write_tensor(
            "embed_tokens.weight", (model["vocab_size"], hidden_size)
        )
    else:
        writer.write_tensor(
            "lm_head.weight",
            (model["vocab_size"], hidden_size),
            transpose=True,
        )
    return writer.elements_written * FP32_BYTES


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert local HuggingFace Gemma4 safetensors to an NNTrainer "
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

    print("NNTrainer Gemma4 streaming FP32 converter")
    print(f"  Source: {model_path}")
    print(f"  Target: {output_path}")
    print(f"  Layers: {model['num_layers']}")
    print(f"  Experts per layer: {model['num_experts']}")
    print(f"  Chunk size: {args.chunk_size_mib} MiB")

    try:
        with SafetensorsSource(model_path) as source:
            with partial_path.open("wb") as output:
                expected_size = write_gemma4(
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
