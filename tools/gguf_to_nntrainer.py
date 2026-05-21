#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file    gguf_to_nntrainer.py
# @date    02 April 2026
# @brief   GGUF to NNTrainer weight converter for Bonsai Q1_0 models
# @author  Jijoong Moon <jijoong.moon@samsung.com>
"""
GGUF to NNTrainer weight converter for Bonsai Q1_0 models.

Reads a GGUF file (e.g., Bonsai-1.7B-Q1_0-g128.gguf), extracts model
architecture metadata and Q1_0 quantized weights, and writes them in
NNTrainer's binary weight format (.bin) for use with CausalLM application.

Also generates config.json, nntr_config.json, and generation_config.json.

Usage:
    python gguf_to_nntrainer.py --gguf /path/to/Bonsai-1.7B-Q1_0-g128.gguf \
                                --output /path/to/output_dir

Requirements:
    pip install gguf numpy

The output directory will contain:
    - config.json            (HuggingFace-style model config)
    - generation_config.json (generation parameters)
    - nntr_config.json       (NNTrainer-specific config)
    - model.bin              (binary weights in NNTrainer order)
"""

import argparse
import json
import os
import struct
import sys
from collections import OrderedDict

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: 'gguf' package is required. Install with: pip install gguf")
    sys.exit(1)


# ============================================================
# GGUF tensor name → NNTrainer weight order mapping
# ============================================================

# GGUF uses standardized tensor names like:
#   token_embd.weight, blk.0.attn_q.weight, blk.0.ffn_up.weight, etc.
# NNTrainer CausalLM expects weights in a specific sequential order.

GGUF_TO_HF_NAME = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Per-block tensor name patterns (replace {i} with layer index)
GGUF_BLOCK_PATTERNS = {
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
}

# NNTrainer CausalLM weight save order per layer:
LAYER_WEIGHT_ORDER = [
    "input_layernorm.weight",           # attn norm
    "self_attn.q_proj.weight",          # Q projection (transposed)
    "self_attn.q_norm.weight",          # Q norm (Qwen3)
    "self_attn.k_proj.weight",          # K projection (transposed)
    "self_attn.k_norm.weight",          # K norm (Qwen3)
    "self_attn.v_proj.weight",          # V projection (transposed)
    "self_attn.o_proj.weight",          # O projection (transposed)
    "post_attention_layernorm.weight",  # FFN norm
    "self_attn.up_proj.weight",         # up (actually mlp.up_proj)
    "self_attn.gate_proj.weight",       # gate (actually mlp.gate_proj)
    "self_attn.down_proj.weight",       # down (actually mlp.down_proj)
]

# Map layer weight names to gguf pattern keys
LAYER_NAME_TO_GGUF = {
    "input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "self_attn.q_norm.weight": "blk.{i}.attn_q_norm.weight",
    "self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "self_attn.k_norm.weight": "blk.{i}.attn_k_norm.weight",
    "self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
    "self_attn.up_proj.weight": "blk.{i}.ffn_up.weight",
    "self_attn.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "self_attn.down_proj.weight": "blk.{i}.ffn_down.weight",
}


def read_gguf_metadata(reader):
    """Extract key metadata fields from GGUF reader."""
    metadata = {}
    for key, field in reader.fields.items():
        try:
            if field.types and len(field.data) > 0:
                # Try to extract the value
                if hasattr(field, 'parts') and len(field.parts) > 0:
                    val = field.parts[-1]
                    if hasattr(val, 'tolist'):
                        val = val.tolist()
                        if len(val) == 1:
                            val = val[0]
                    metadata[key] = val
                else:
                    metadata[key] = field.data.tolist()
        except Exception:
            pass
    return metadata


def dequantize_q1_0_block(block_data):
    """Dequantize a Q1_0 block (18 bytes) to 128 float32 values.

    Block layout:
      - 2 bytes: FP16 scale
      - 16 bytes: 128 bits (1 bit per weight)
    bit=1 → +scale, bit=0 → -scale
    """
    scale_bytes = block_data[:2]
    scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)

    bit_bytes = block_data[2:18]
    weights = np.zeros(128, dtype=np.float32)

    for byte_idx in range(16):
        byte_val = bit_bytes[byte_idx]
        for bit_idx in range(8):
            w_idx = byte_idx * 8 + bit_idx
            bit = (byte_val >> bit_idx) & 1
            weights[w_idx] = scale if bit else -scale

    return weights


def extract_q1_0_raw(tensor_data, shape):
    """Extract raw Q1_0 quantized data as bytes (for native Q1_0 loading).

    Returns the raw block data that can be loaded directly by nntrainer's
    Q1_0_Tensor.
    """
    return bytes(tensor_data)


def dequantize_q1_0_tensor(tensor_data, shape):
    """Dequantize a full Q1_0 tensor to float32.

    Args:
        tensor_data: raw bytes of Q1_0 blocks
        shape: tensor shape (rows, cols)

    Returns:
        numpy float32 array
    """
    if len(shape) == 1:
        total_elements = shape[0]
        rows, cols = 1, shape[0]
    else:
        rows, cols = shape[0], shape[1]
        total_elements = rows * cols

    num_blocks = total_elements // 128
    block_size = 18  # 2 (fp16 scale) + 16 (128 bits)

    result = np.zeros(total_elements, dtype=np.float32)
    data = bytes(tensor_data)

    for b in range(num_blocks):
        block = data[b * block_size:(b + 1) * block_size]
        result[b * 128:(b + 1) * 128] = dequantize_q1_0_block(block)

    return result.reshape(shape)


def get_tensor_data(tensor):
    """Get raw data from a GGUF tensor."""
    return tensor.data.tobytes()


def save_nntrainer_bin(reader, metadata, output_dir, dequantize=True):
    """Save weights in NNTrainer binary format.

    Args:
        reader: GGUFReader instance
        metadata: extracted metadata dict
        output_dir: output directory
        dequantize: if True, dequantize Q1_0 to FP32
    """
    # Build tensor lookup
    tensor_map = {}
    for tensor in reader.tensors:
        tensor_map[tensor.name] = tensor

    # Determine number of layers
    arch = metadata.get("general.architecture", "unknown")
    n_layers_key = f"{arch}.block_count"
    n_layers = metadata.get(n_layers_key, 0)

    if n_layers == 0:
        # Try to infer from tensor names
        for name in tensor_map:
            if name.startswith("blk."):
                idx = int(name.split(".")[1])
                n_layers = max(n_layers, idx + 1)

    print(f"Architecture: {arch}")
    print(f"Number of layers: {n_layers}")
    print(f"Total tensors: {len(tensor_map)}")

    dtype_str = "float32" if dequantize else "q1_0"
    output_file = os.path.join(output_dir,
                               f"model_{dtype_str}.bin")

    weight_count = 0
    total_bytes = 0

    with open(output_file, "wb") as f:

        # 1. Save embedding
        if "token_embd.weight" in tensor_map:
            t = tensor_map["token_embd.weight"]
            data = get_tensor_data(t)
            if dequantize and t.tensor_type != 0:  # Not F32
                arr = dequantize_q1_0_tensor(data, list(t.shape))
                data = arr.astype(np.float32).tobytes()
            else:
                arr = np.frombuffer(data, dtype=np.float32)
            f.write(data)
            weight_count += 1
            total_bytes += len(data)
            print(f"  [embedding] token_embd.weight: shape={list(t.shape)}, "
                  f"type={t.tensor_type}, size={len(data)}")

        # 2. Save each layer
        for layer_idx in range(n_layers):
            for weight_name in LAYER_WEIGHT_ORDER:
                gguf_key = LAYER_NAME_TO_GGUF[weight_name].format(i=layer_idx)

                if gguf_key not in tensor_map:
                    # Some weights are optional (e.g., q_norm, k_norm)
                    continue

                t = tensor_map[gguf_key]
                data = get_tensor_data(t)

                # Check if this is a projection weight that needs transpose
                is_projection = any(p in weight_name for p in
                                    ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "up_proj", "gate_proj", "down_proj"])

                if dequantize and t.tensor_type != 0:
                    arr = dequantize_q1_0_tensor(data, list(t.shape))
                    if is_projection and arr.ndim == 2:
                        arr = arr.T.copy()  # Transpose for NNTrainer
                    data = arr.astype(np.float32).tobytes()
                elif t.tensor_type == 0:  # Already F32
                    arr = np.frombuffer(data, dtype=np.float32).reshape(t.shape)
                    if is_projection and arr.ndim == 2:
                        arr = arr.T.copy()
                        data = arr.tobytes()

                f.write(data)
                weight_count += 1
                total_bytes += len(data)

            if (layer_idx + 1) % 4 == 0 or layer_idx == n_layers - 1:
                print(f"  [layer {layer_idx}] saved")

        # 3. Save final norm
        if "output_norm.weight" in tensor_map:
            t = tensor_map["output_norm.weight"]
            data = get_tensor_data(t)
            if dequantize and t.tensor_type != 0:
                arr = dequantize_q1_0_tensor(data, list(t.shape))
                data = arr.astype(np.float32).tobytes()
            f.write(data)
            weight_count += 1
            total_bytes += len(data)
            print(f"  [final_norm] output_norm.weight")

        # 4. Save lm_head
        if "output.weight" in tensor_map:
            t = tensor_map["output.weight"]
            data = get_tensor_data(t)
            if dequantize and t.tensor_type != 0:
                arr = dequantize_q1_0_tensor(data, list(t.shape))
                arr = arr.T.copy()  # Transpose for NNTrainer
                data = arr.astype(np.float32).tobytes()
            elif t.tensor_type == 0:
                arr = np.frombuffer(data, dtype=np.float32).reshape(t.shape)
                arr = arr.T.copy()
                data = arr.tobytes()
            f.write(data)
            weight_count += 1
            total_bytes += len(data)
            print(f"  [lm_head] output.weight")
        elif metadata.get(f"{arch}.tie_word_embeddings", False):
            print("  [lm_head] tied to embedding (no separate weight)")

    print(f"\nSaved {weight_count} weights, total {total_bytes / 1e6:.1f} MB")
    print(f"Output: {output_file}")
    return output_file


def generate_configs(metadata, output_dir, model_bin_name, tokenizer_path=""):
    """Generate config.json, generation_config.json, and nntr_config.json."""
    arch = metadata.get("general.architecture", "qwen3")

    # Extract architecture parameters
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": arch,
        "vocab_size": metadata.get(f"{arch}.vocab_size",
                                   metadata.get("tokenizer.ggml.tokens", 0)),
        "hidden_size": metadata.get(f"{arch}.embedding_length", 0),
        "intermediate_size": metadata.get(f"{arch}.feed_forward_length", 0),
        "num_hidden_layers": metadata.get(f"{arch}.block_count", 0),
        "num_attention_heads": metadata.get(f"{arch}.attention.head_count", 0),
        "num_key_value_heads": metadata.get(
            f"{arch}.attention.head_count_kv",
            metadata.get(f"{arch}.attention.head_count", 0)),
        "head_dim": metadata.get(f"{arch}.attention.key_length",
                                 metadata.get(f"{arch}.attention.value_length", 0)),
        "max_position_embeddings": metadata.get(f"{arch}.context_length", 32768),
        "rope_theta": metadata.get(f"{arch}.rope.freq_base", 1000000),
        "rms_norm_eps": metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
        "tie_word_embeddings": metadata.get(f"{arch}.tie_word_embeddings", False),
        "sliding_window": None,
    }

    # Handle vocab_size from tokenizer tokens array
    if isinstance(config["vocab_size"], list):
        config["vocab_size"] = len(config["vocab_size"])

    # Handle head_dim fallback
    if config["head_dim"] == 0 and config["hidden_size"] > 0 and config["num_attention_heads"] > 0:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Generation config
    gen_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": 512,
        "eos_token_id": [
            metadata.get("tokenizer.ggml.eos_token_id", 151645)
        ],
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    # NNTrainer config
    nntr_config = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": model_bin_name,
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": 128,
        "init_seq_len": 512,
        "max_seq_len": 1024,
        "batch_size": 1,
        "tokenizer_file": tokenizer_path if tokenizer_path else
            os.path.join(output_dir, "tokenizer.json"),
        "sample_input": "<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    }
    with open(os.path.join(output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_config, f, indent=2)

    print(f"\nGenerated configs in {output_dir}/")
    print(f"  Architecture: {config.get('architectures', ['unknown'])[0]}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Heads: {config['num_attention_heads']} "
          f"(KV: {config['num_key_value_heads']})")
    print(f"  Vocab: {config['vocab_size']}")
    print(f"  Head dim: {config['head_dim']}")


def dump_metadata(reader):
    """Print all GGUF metadata for inspection."""
    print("=" * 60)
    print("GGUF Metadata")
    print("=" * 60)
    metadata = read_gguf_metadata(reader)
    for key, val in sorted(metadata.items()):
        if isinstance(val, (list, np.ndarray)) and len(val) > 10:
            print(f"  {key}: [{type(val).__name__}, len={len(val)}]")
        else:
            print(f"  {key}: {val}")

    print(f"\n{'=' * 60}")
    print("Tensors")
    print("=" * 60)
    for t in reader.tensors:
        type_names = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0",
                      12: "Q4_K", 14: "Q6_K", 41: "Q1_0"}
        ttype = type_names.get(t.tensor_type, f"type_{t.tensor_type}")
        size_bytes = len(t.data.tobytes())
        print(f"  {t.name:50s}  shape={list(t.shape):20s}  "
              f"type={ttype:6s}  size={size_bytes:>10,}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF Q1_0 model to NNTrainer format")
    parser.add_argument("--gguf", required=True,
                        help="Path to GGUF file")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--dump-only", action="store_true",
                        help="Only dump metadata without converting")
    parser.add_argument("--dequantize", action="store_true", default=True,
                        help="Dequantize Q1_0 to FP32 (default: True)")
    parser.add_argument("--no-dequantize", action="store_true",
                        help="Keep Q1_0 format (native, experimental)")
    parser.add_argument("--tokenizer", default="",
                        help="Path to tokenizer.json")
    args = parser.parse_args()

    if args.no_dequantize:
        args.dequantize = False

    print(f"Reading GGUF: {args.gguf}")
    reader = GGUFReader(args.gguf)

    metadata = dump_metadata(reader)

    if args.dump_only:
        return

    os.makedirs(args.output, exist_ok=True)

    # Save weights
    model_bin = save_nntrainer_bin(reader, metadata, args.output,
                                   dequantize=args.dequantize)

    # Generate configs
    model_bin_name = os.path.basename(model_bin)
    generate_configs(metadata, args.output, model_bin_name, args.tokenizer)

    print(f"\n{'=' * 60}")
    print("Done! To run with NNTrainer CausalLM:")
    print(f"  ./nntrainer_causal_lm {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
