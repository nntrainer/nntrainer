# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>

## @file weight_converter.py
## @brief weight conversion script for gemma4 text model
## @author Joonseok Oh <jrock.oh@samsung.com>
##
## Converts a HuggingFace Gemma4 text model into the nntrainer weight format,
## supporting both the legacy binary (.bin) layout and the safetensors layout.
##
## Both paths share a single ordered weight walker so the two formats stay in
## sync. The .bin path writes the payload positionally (matching nntrainer's
## binary save/load order); the safetensors path keys each tensor by its
## nntrainer weight name (`<layer_name>:<weight_name>`), which is how the
## nntrainer safetensors loader matches tensors back to layers.
##
## Gemma4 specifics handled here:
## - 35 text layers with interleaved sliding / full attention
## - KV sharing for the last 20 layers (layer >= 15): wk / wv / k_norm are not
##   emitted, matching nntrainer's createSharedAttention()
## - Double-wide MLP for KV-shared layers (shape is taken from the source tensor)
## - Per-layer input embedding / projection / norm (global, emitted with layer 0)
## - Tied word embeddings: the lm head shares embedding0, so the safetensors
##   path emits no separate output_of_quick_ai entry, while the .bin path keeps
##   the trailing duplicate that nntrainer's binary lm-head save expects.

import argparse
import glob
import json
import os
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}

# nntrainer internal weight names per layer type (the part after ':').
SUFFIX_WEIGHT = "weight"            # fully_connected
SUFFIX_GAMMA = "gamma"             # rms_norm / reshaped_rms_norm
SUFFIX_EMBEDDING = "Embedding"     # embedding / tie_word_embeddings
SUFFIX_SCALAR = "scalar_multiplier"  # scalar_multiply(use_weight=true)


def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert a torch tensor to a contiguous numpy array of the given dtype.

    Accepts either a torch tensor or a LazyTensor; the latter loads its data
    from the safetensors file only at this point, so only one weight is held
    in memory at a time.
    """
    if hasattr(tensor, "materialize"):
        tensor = tensor.materialize()
    if transpose:
        tensor = tensor.permute(1, 0)

    # Upcast to float32 once (handles bfloat16) and reuse that buffer for the
    # numpy view; only copy again if the requested numpy dtype differs or the
    # data is not contiguous. This avoids a redundant full-size copy on the
    # large embedding tensors.
    tensor = tensor.detach()
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    arr = tensor.contiguous().cpu().numpy()

    np_dtype = np.dtype(dtype)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    return np.ascontiguousarray(arr)


def get_tie_word_embeddings(text_config, config):
    """Return whether the model ties the lm head to the input embedding."""
    if hasattr(text_config, "tie_word_embeddings"):
        return bool(text_config.tie_word_embeddings)
    return bool(getattr(config, "tie_word_embeddings", True))


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"

    if output_name.endswith(".safetensors"):
        return output_name

    return output_name + ".safetensors"


def make_param_resolver(params):
    """Build a resolver that finds HF keys regardless of the model wrapper prefix.

    Gemma4 may load as a multimodal model (keys under
    `model.language_model.`), a bare text model (`model.`) or with no prefix at
    all. The resolver tries each known prefix in turn.
    """
    prefixes = ["model.language_model.", "language_model.", "model.", ""]

    def resolve(relative_key, required=True):
        for prefix in prefixes:
            full_key = prefix + relative_key
            if full_key in params:
                return params[full_key]
        if required:
            raise KeyError(
                f"Could not find '{relative_key}' under any known prefix "
                f"{prefixes}"
            )
        return None

    return resolve


class LazyTensor:
    """A handle to one tensor inside a safetensors file.

    Exposes the shape without reading the payload (so the safetensors header
    can be built cheaply) and loads the actual data only on materialize().
    """

    def __init__(self, handle, key):
        self._handle = handle
        self._key = key

    @property
    def shape(self):
        return tuple(self._handle.get_slice(self._key).get_shape())

    def materialize(self):
        return self._handle.get_tensor(self._key)


class SafetensorsState:
    """A lazy, dict-like view over one or more safetensors files.

    Used instead of loading the whole HuggingFace model into RAM: only the
    tensor currently being converted is read from disk, keeping peak memory at
    roughly a single weight.
    """

    def __init__(self, files):
        self._handles = []
        self._key_to_handle = {}
        # `safetensors` is imported lazily so the binary path never requires it.
        from safetensors import safe_open

        for path in files:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles.append(handle)
            for key in handle.keys():
                self._key_to_handle[key] = handle

    def __contains__(self, key):
        return key in self._key_to_handle

    def __getitem__(self, key):
        return LazyTensor(self._key_to_handle[key], key)

    def keys(self):
        return self._key_to_handle.keys()


def load_model_state(model_path):
    """Return a (state, description) pair to read weights from.

    Prefers reading a local safetensors checkpoint lazily (no full model in
    RAM). Falls back to loading the HuggingFace model when no local
    safetensors file is found (e.g. a hub id or a .bin-only checkpoint).
    """
    if os.path.isdir(model_path):
        files = sorted(
            f for f in glob.glob(os.path.join(model_path, "*.safetensors"))
            if not f.endswith(".index.json")
        )
        if files:
            return SafetensorsState(files), (
                f"{len(files)} local safetensors file(s) (lazy)"
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model.state_dict(), "HuggingFace model state_dict"


def iter_gemma4_weight_specs(params, config):
    """Yield (nntr_name, suffix, torch_tensor, transpose) in nntrainer order.

    The ordering matches nntrainer's binary save/load order. The names and
    suffixes match the nntrainer layer graph (see gemma4_causallm.cpp), so the
    same walk drives both the .bin and the safetensors output.

    Tensors are yielded as references into the model state dict, so this holds
    no extra memory; callers convert each tensor to numpy one at a time and
    release it before moving on. The tied lm head (output_of_quick_ai) is
    intentionally excluded here; the two output paths handle it themselves.
    """
    text_config = config.text_config
    n_layers = text_config.num_hidden_layers
    num_kv_shared_layers = text_config.num_kv_shared_layers
    first_kv_shared_layer_idx = n_layers - num_kv_shared_layers

    resolve = make_param_resolver(params)

    def is_kv_shared_layer(layer_idx):
        return layer_idx >= first_kv_shared_layer_idx

    # 1. Main (input) embedding -> embedding0
    yield ("embedding0", SUFFIX_EMBEDDING, resolve("embed_tokens.weight"), False)

    # 2. Decoder layers
    for layer_idx in range(n_layers):
        lp = f"layers.{layer_idx}."
        is_kv_shared = is_kv_shared_layer(layer_idx)

        yield (f"layer{layer_idx}_attention_norm", SUFFIX_GAMMA,
               resolve(f"{lp}input_layernorm.weight"), False)

        yield (f"layer{layer_idx}_wq", SUFFIX_WEIGHT,
               resolve(f"{lp}self_attn.q_proj.weight"), True)
        yield (f"layer{layer_idx}_q_norm", SUFFIX_GAMMA,
               resolve(f"{lp}self_attn.q_norm.weight"), False)

        # KV-shared layers reuse an earlier layer's K/V, so wk / wv / k_norm
        # are not part of the nntrainer graph for them.
        if not is_kv_shared:
            yield (f"layer{layer_idx}_wk", SUFFIX_WEIGHT,
                   resolve(f"{lp}self_attn.k_proj.weight"), True)
            yield (f"layer{layer_idx}_k_norm", SUFFIX_GAMMA,
                   resolve(f"{lp}self_attn.k_norm.weight"), False)
            yield (f"layer{layer_idx}_wv", SUFFIX_WEIGHT,
                   resolve(f"{lp}self_attn.v_proj.weight"), True)

        yield (f"layer{layer_idx}_attention_out", SUFFIX_WEIGHT,
               resolve(f"{lp}self_attn.o_proj.weight"), True)

        yield (f"layer{layer_idx}_post_attention_norm", SUFFIX_GAMMA,
               resolve(f"{lp}post_attention_layernorm.weight"), False)
        yield (f"layer{layer_idx}_pre_ffn_norm", SUFFIX_GAMMA,
               resolve(f"{lp}pre_feedforward_layernorm.weight"), False)

        yield (f"layer{layer_idx}_ffn_gate", SUFFIX_WEIGHT,
               resolve(f"{lp}mlp.gate_proj.weight"), True)
        yield (f"layer{layer_idx}_ffn_up", SUFFIX_WEIGHT,
               resolve(f"{lp}mlp.up_proj.weight"), True)
        yield (f"layer{layer_idx}_ffn_down", SUFFIX_WEIGHT,
               resolve(f"{lp}mlp.down_proj.weight"), True)

        yield (f"layer{layer_idx}_post_ffn_norm", SUFFIX_GAMMA,
               resolve(f"{lp}post_feedforward_layernorm.weight"), False)

        yield (f"layer{layer_idx}_per_layer_input_gate", SUFFIX_WEIGHT,
               resolve(f"{lp}per_layer_input_gate.weight"), True)

        # Global per-layer input weights live in the graph next to layer 0.
        if layer_idx == 0:
            yield ("per_layer_input_embedding", SUFFIX_EMBEDDING,
                   resolve("embed_tokens_per_layer.weight"), False)
            yield ("per_layer_input_projection", SUFFIX_WEIGHT,
                   resolve("per_layer_model_projection.weight"), True)
            yield ("per_layer_projection_norm", SUFFIX_GAMMA,
                   resolve("per_layer_projection_norm.weight"), False)

        yield (f"layer{layer_idx}_per_layer_input_proj", SUFFIX_WEIGHT,
               resolve(f"{lp}per_layer_projection.weight"), True)
        yield (f"layer{layer_idx}_post_per_layer_input_norm", SUFFIX_GAMMA,
               resolve(f"{lp}post_per_layer_input_norm.weight"), False)
        yield (f"layer{layer_idx}_layer_scalar", SUFFIX_SCALAR,
               resolve(f"{lp}layer_scalar"), False)

    # 3. Final norm
    yield ("output_norm", SUFFIX_GAMMA, resolve("norm.weight"), False)


def lm_head_spec(params, tie_word_embeddings):
    """Return the (name, suffix, tensor, transpose) spec for the lm head.

    With tied embeddings the lm head shares embedding0, so the binary path
    re-emits the shared embedding (no transpose); otherwise it writes the
    dedicated, transposed lm_head weight.
    """
    resolve = make_param_resolver(params)
    if tie_word_embeddings:
        return ("output_of_quick_ai", SUFFIX_EMBEDDING,
                resolve("embed_tokens.weight"), False)
    return ("output_of_quick_ai", SUFFIX_WEIGHT,
            resolve("lm_head.weight"), True)


def _transposed_shape(tensor, transpose):
    """Return the numpy output shape for a tensor, accounting for transpose."""
    shape = list(tensor.shape)
    if transpose:
        shape = [shape[1], shape[0]]
    return shape


def save_gemma4_bin(params, config, dtype, file, tie_word_embeddings):
    """Write Gemma4 weights as the nntrainer binary (.bin) layout.

    Streams one tensor at a time: each weight is converted to numpy, written,
    then released, so peak memory stays at the model plus a single tensor.
    """
    total_bytes = 0
    count = 0
    for _name, _suffix, tensor, transpose in iter_gemma4_weight_specs(
            params, config):
        arr = tensor_to_numpy(tensor, dtype, transpose)
        arr.tofile(file)
        total_bytes += arr.nbytes
        count += 1
        del arr

    # lm head. With tied embeddings nntrainer's binary lm-head save re-emits
    # the shared embedding, so keep that trailing duplicate here.
    _name, _suffix, tensor, transpose = lm_head_spec(params, tie_word_embeddings)
    arr = tensor_to_numpy(tensor, dtype, transpose)
    arr.tofile(file)
    total_bytes += arr.nbytes
    count += 1
    del arr

    print(f"Saved binary tensors: {count}")
    print(f"Total bytes: {total_bytes:,} "
          f"({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    return total_bytes


def save_gemma4_safetensors(params, config, dtype, output_path,
                            tie_word_embeddings):
    """Write Gemma4 weights as a safetensors file keyed by nntrainer names.

    The safetensors layout is [8-byte header length][header JSON][raw data].
    The header only needs each tensor's shape, so it is built from tensor
    metadata first (no materialization). The raw data is then streamed one
    tensor at a time, keeping peak memory at the model plus a single tensor
    instead of holding a full numpy copy and a full byte-buffer copy at once.
    """
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    itemsize = np.dtype(dtype).itemsize

    specs = list(iter_gemma4_weight_specs(params, config))

    # With tied embeddings the lm head shares embedding0's tensor, so nntrainer
    # stores a single deduped entry and no separate output_of_quick_ai is added.
    if not tie_word_embeddings:
        specs.append(lm_head_spec(params, tie_word_embeddings))

    # Pass 1: build the header from shapes only (no tensor data touched).
    offset = 0
    tensor_meta = {}
    for name, suffix, tensor, transpose in specs:
        shape = _transposed_shape(tensor, transpose)
        nbytes = itemsize
        for dim in shape:
            nbytes *= dim
        tensor_meta[f"{name}:{suffix}"] = {
            "dtype": safetensors_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header = {"__metadata__": {"format": "pt"}}
    header.update(tensor_meta)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    # Pass 2: write header, then stream each tensor's bytes and release it.
    with open(output_path, "wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)

        for _name, _suffix, tensor, transpose in specs:
            arr = tensor_to_numpy(tensor, dtype, transpose)
            arr.tofile(output_file)
            del arr

    print(f"Saved safetensors: {output_path}")
    print(f"Tensors: {len(specs)}")
    print(f"Tensor data size: {offset / 1e9:.2f} GB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Gemma4 text model weights to nntrainer format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=".",
        help="HuggingFace model path or local Gemma4 model directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_gemma4_fp32.bin",
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
    """Convert Gemma4 HuggingFace weights to nntrainer weight format."""
    args = parse_args()

    print(f"Loading Gemma4 model from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Read weights lazily from the local safetensors checkpoint when possible,
    # so the full (multimodal) model is never resident in RAM. Each tensor is
    # loaded and upcast to float32 only while it is being written, keeping peak
    # memory at roughly a single weight. Falls back to a HuggingFace model load
    # for hub ids or non-safetensors checkpoints.
    params, source = load_model_state(args.model_path)
    print(f"Weight source: {source}")

    text_config = config.text_config
    tie_word_embeddings = get_tie_word_embeddings(text_config, config)

    print("\nModel configuration:")
    print(f"  Text layers: {text_config.num_hidden_layers}")
    print(f"  Hidden size: {text_config.hidden_size}")
    print(f"  Vocab size: {text_config.vocab_size}")
    print(f"  KV shared layers: {text_config.num_kv_shared_layers}")
    print(f"  Tie word embeddings: {tie_word_embeddings}")
    print(f"  Output dtype: {args.data_type}")

    if args.safetensors:
        output_name = get_safetensors_output_name(args.output_name)
        save_gemma4_safetensors(
            params, config, args.data_type, output_name, tie_word_embeddings
        )
        return

    with open(args.output_name, "wb") as output_file:
        save_gemma4_bin(
            params, config, args.data_type, output_file, tie_word_embeddings
        )
    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()
