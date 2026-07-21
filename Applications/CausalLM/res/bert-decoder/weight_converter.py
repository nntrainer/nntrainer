# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief Weight conversion script for the BERT cross-attention decoder
##        (decoder half of a VisionEncoderDecoder checkpoint).
## @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import json
import struct

import numpy as np
from safetensors.torch import load_file


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}

DEC_LAYERS = 4  # BERT decoder


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from res/qwen3/qwen3-0.6b/weight_converter.py)
# ---------------------------------------------------------------------------

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
    print(f"Tensor data size: {offset / 1e9:.4f} GB")


# ---------------------------------------------------------------------------
# Decoder weight collection
# Order MUST match BertDecoder layer creation order.
#
# 1. word_embeddings     [30522, 256]
# 2. position_embeddings [512, 256]
# 3. token_type_embeddings [2, 256]
# 4. embeddings.LayerNorm weight, bias
# 5. For each of 4 decoder layers:
#      self: query(T),qb, key(T),kb, value(T),vb
#      attention.output.dense(T), bias
#      attention.output.LayerNorm weight, bias
#      cross: query(T),qb, key(T),kb, value(T),vb
#      crossattention.output.dense(T), bias
#      crossattention.output.LayerNorm weight, bias
#      intermediate.dense(T), bias
#      output.dense(T), bias
#      output.LayerNorm weight, bias
# 6. LM head:
#      cls.predictions.transform.dense(T), bias
#      cls.predictions.transform.LayerNorm weight, bias
#      cls.predictions.bias [30522]
#    (word-embedding projection is TIED — do NOT re-save the 30522×256 matrix)
# ---------------------------------------------------------------------------

def collect_decoder(sd, dtype):
    """Return ordered list of (nntr_name, ndarray) for the decoder file."""
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    bp = "decoder.bert."

    # 1–4. Embeddings
    # NOTE tensor names must match the runtime weight names exactly: the
    # safetensors loader matches by name and silently skips misses. The
    # embedding_layer names its lookup table "Embedding" (not "weight").
    add("word_emb:Embedding",
        sd[f"{bp}embeddings.word_embeddings.weight"])
    add("pos_emb:Embedding",
        sd[f"{bp}embeddings.position_embeddings.weight"])
    add("type_emb:Embedding",
        sd[f"{bp}embeddings.token_type_embeddings.weight"])
    add("emb_ln:gamma",
        sd[f"{bp}embeddings.LayerNorm.weight"])
    add("emb_ln:beta",
        sd[f"{bp}embeddings.LayerNorm.bias"])

    # 5. Decoder layers
    for i in range(DEC_LAYERS):
        lp = f"{bp}encoder.layer.{i}."
        pfx = f"dec_layer{i}"

        # Self-attention QKV
        add(f"{pfx}_self_q:weight",
            sd[f"{lp}attention.self.query.weight"], transpose=True)
        add(f"{pfx}_self_q:bias",
            sd[f"{lp}attention.self.query.bias"])
        add(f"{pfx}_self_k:weight",
            sd[f"{lp}attention.self.key.weight"], transpose=True)
        add(f"{pfx}_self_k:bias",
            sd[f"{lp}attention.self.key.bias"])
        add(f"{pfx}_self_v:weight",
            sd[f"{lp}attention.self.value.weight"], transpose=True)
        add(f"{pfx}_self_v:bias",
            sd[f"{lp}attention.self.value.bias"])

        # Self-attention output dense + LayerNorm
        add(f"{pfx}_self_out:weight",
            sd[f"{lp}attention.output.dense.weight"], transpose=True)
        add(f"{pfx}_self_out:bias",
            sd[f"{lp}attention.output.dense.bias"])
        add(f"{pfx}_self_ln:gamma",
            sd[f"{lp}attention.output.LayerNorm.weight"])
        add(f"{pfx}_self_ln:beta",
            sd[f"{lp}attention.output.LayerNorm.bias"])

        # Cross-attention QKV
        add(f"{pfx}_cross_q:weight",
            sd[f"{lp}crossattention.self.query.weight"], transpose=True)
        add(f"{pfx}_cross_q:bias",
            sd[f"{lp}crossattention.self.query.bias"])
        add(f"{pfx}_cross_k:weight",
            sd[f"{lp}crossattention.self.key.weight"], transpose=True)
        add(f"{pfx}_cross_k:bias",
            sd[f"{lp}crossattention.self.key.bias"])
        add(f"{pfx}_cross_v:weight",
            sd[f"{lp}crossattention.self.value.weight"], transpose=True)
        add(f"{pfx}_cross_v:bias",
            sd[f"{lp}crossattention.self.value.bias"])

        # Cross-attention output dense + LayerNorm
        add(f"{pfx}_cross_out:weight",
            sd[f"{lp}crossattention.output.dense.weight"], transpose=True)
        add(f"{pfx}_cross_out:bias",
            sd[f"{lp}crossattention.output.dense.bias"])
        add(f"{pfx}_cross_ln:gamma",
            sd[f"{lp}crossattention.output.LayerNorm.weight"])
        add(f"{pfx}_cross_ln:beta",
            sd[f"{lp}crossattention.output.LayerNorm.bias"])

        # FFN: intermediate + output
        add(f"{pfx}_ffn_inter:weight",
            sd[f"{lp}intermediate.dense.weight"], transpose=True)
        add(f"{pfx}_ffn_inter:bias",
            sd[f"{lp}intermediate.dense.bias"])
        add(f"{pfx}_ffn_out:weight",
            sd[f"{lp}output.dense.weight"], transpose=True)
        add(f"{pfx}_ffn_out:bias",
            sd[f"{lp}output.dense.bias"])
        add(f"{pfx}_ffn_ln:gamma",
            sd[f"{lp}output.LayerNorm.weight"])
        add(f"{pfx}_ffn_ln:beta",
            sd[f"{lp}output.LayerNorm.bias"])

    # 6. LM head (word-embedding projection is TIED — not re-saved)
    cp = "decoder.cls."
    add("lmhead_dense:weight",
        sd[f"{cp}predictions.transform.dense.weight"], transpose=True)
    add("lmhead_dense:bias",
        sd[f"{cp}predictions.transform.dense.bias"])
    add("lmhead_ln:gamma",
        sd[f"{cp}predictions.transform.LayerNorm.weight"])
    add("lmhead_ln:beta",
        sd[f"{cp}predictions.transform.LayerNorm.bias"])
    # weight layer "lm_head_bias/weights" with weight_name "lmhead_bias"
    add("lm_head_bias/weights:lmhead_bias",
        sd[f"{cp}predictions.bias"])

    return weights


# ---------------------------------------------------------------------------
# Binary save helpers
# ---------------------------------------------------------------------------

def save_bin(weights, output_path):
    """Write weights to nntrainer raw binary file.

    Arrays are already in the target dtype (converted by tensor_to_numpy).
    """
    with open(output_path, "wb") as f:
        for _name, arr in weights:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            arr.tofile(f)
    total_bytes = sum(arr.nbytes for _, arr in weights)
    print(f"Saved binary: {output_path}")
    print(f"Tensor data size: {total_bytes / 1e9:.4f} GB")


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert the BERT cross-attention decoder of a source "
        "safetensors checkpoint to nntrainer format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.safetensors",
        help="Path to the source model.safetensors",
    )
    parser.add_argument(
        "--decoder_output",
        type=str,
        default="./nntr_bert_decoder_fp32.bin",
        help="Output path for decoder weight file",
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
        help="Save in safetensors format instead of binary",
    )
    return parser.parse_args()


def main():
    """Convert the BERT decoder to an nntrainer weight file."""
    args = parse_args()

    print(f"Loading checkpoint: {args.model_path}")
    sd = load_file(args.model_path)
    print(f"Checkpoint loaded — {len(sd)} tensors total.")

    dtype = args.data_type

    dec_weights = collect_decoder(sd, dtype)
    print(f"\nDecoder tensors: {len(dec_weights)}")
    assert len(dec_weights) == 114, (
        f"DECODER COUNT MISMATCH: got {len(dec_weights)}, expected 114"
    )

    if args.safetensors:
        out = get_safetensors_output_name(args.decoder_output)
        save_safetensors(dec_weights, out, dtype)
    else:
        save_bin(dec_weights, args.decoder_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
