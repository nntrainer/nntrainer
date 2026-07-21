# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief Weight conversion script for the SigLIP2 vision encoder
##        (encoder half of a VisionEncoderDecoder checkpoint).
## @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import json
import struct

import numpy as np
from safetensors.torch import load_file


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}

ENC_LAYERS = 12  # SigLIP2 ViT-B/16


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
# Encoder weight collection
# Order MUST match Siglip2VisionEncoder::constructModel (Task 3).
#
# 1.  patch_embed conv weight  [768, 3, 16, 16]  OIHW — no transpose
# 2.  patch_embed bias         [768]
# 3.  position_embedding       [196, 768] → reshaped [1,1,196,768]
# 4.  For each of 12 encoder layers:
#       layer_norm1 weight, bias
#       q weight(T), bias
#       k weight(T), bias
#       v weight(T), bias
#       out_proj weight(T), bias
#       layer_norm2 weight, bias
#       fc1 weight(T), bias
#       fc2 weight(T), bias
# 5.  post_layernorm weight, bias
# 6.  enc_to_dec_proj weight(T), bias
# ---------------------------------------------------------------------------

def collect_encoder(sd, dtype):
    """Return ordered list of (nntr_name, ndarray) for the encoder file."""
    weights = []

    def add(name, tensor, transpose=False):
        arr = tensor_to_numpy(tensor, dtype, transpose=transpose)
        weights.append((name, arr))

    vp = "encoder.vision_model."

    # 1–2. Patch embedding (Conv2D — keep OIHW, no transpose)
    # NOTE tensor names must match the runtime weight names exactly: the
    # safetensors loader matches by name and silently skips misses. conv2d
    # names its kernel "filter" (not "weight"); the pos_embedding weight
    # layer names its tensor after its weight_name property.
    add("patch_embed_conv:filter",
        sd[f"{vp}embeddings.patch_embedding.weight"],
        transpose=False)
    add("patch_embed_conv:bias",
        sd[f"{vp}embeddings.patch_embedding.bias"])

    # 3. Position embedding [196,768] → [1,1,196,768]
    pos = sd[f"{vp}embeddings.position_embedding.weight"]  # [196, 768]
    pos_reshaped = pos.unsqueeze(0).unsqueeze(0)           # [1, 1, 196, 768]
    add("pos_embedding:pos_embedding", pos_reshaped, transpose=False)

    # 4. Transformer layers
    for i in range(ENC_LAYERS):
        lp = f"{vp}encoder.layers.{i}."
        pfx = f"enc_layer{i}"

        add(f"{pfx}_ln1:gamma",   sd[f"{lp}layer_norm1.weight"])
        add(f"{pfx}_ln1:beta",    sd[f"{lp}layer_norm1.bias"])

        add(f"{pfx}_wq:weight",   sd[f"{lp}self_attn.q_proj.weight"], transpose=True)
        add(f"{pfx}_wq:bias",     sd[f"{lp}self_attn.q_proj.bias"])

        add(f"{pfx}_wk:weight",   sd[f"{lp}self_attn.k_proj.weight"], transpose=True)
        add(f"{pfx}_wk:bias",     sd[f"{lp}self_attn.k_proj.bias"])

        add(f"{pfx}_wv:weight",   sd[f"{lp}self_attn.v_proj.weight"], transpose=True)
        add(f"{pfx}_wv:bias",     sd[f"{lp}self_attn.v_proj.bias"])

        add(f"{pfx}_out:weight",  sd[f"{lp}self_attn.out_proj.weight"], transpose=True)
        add(f"{pfx}_out:bias",    sd[f"{lp}self_attn.out_proj.bias"])

        add(f"{pfx}_ln2:gamma",   sd[f"{lp}layer_norm2.weight"])
        add(f"{pfx}_ln2:beta",    sd[f"{lp}layer_norm2.bias"])

        add(f"{pfx}_fc1:weight",  sd[f"{lp}mlp.fc1.weight"], transpose=True)
        add(f"{pfx}_fc1:bias",    sd[f"{lp}mlp.fc1.bias"])

        add(f"{pfx}_fc2:weight",  sd[f"{lp}mlp.fc2.weight"], transpose=True)
        add(f"{pfx}_fc2:bias",    sd[f"{lp}mlp.fc2.bias"])

    # 5. Post-LayerNorm
    add("post_ln:gamma", sd[f"{vp}post_layernorm.weight"])
    add("post_ln:beta",  sd[f"{vp}post_layernorm.bias"])

    # 6. Encoder-to-decoder projection (written with encoder file)
    add("enc_to_dec_proj:weight", sd["enc_to_dec_proj.weight"], transpose=True)
    add("enc_to_dec_proj:bias",   sd["enc_to_dec_proj.bias"])

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
        description="Convert the SigLIP2 vision encoder of a source "
        "safetensors checkpoint to nntrainer format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.safetensors",
        help="Path to the source model.safetensors",
    )
    parser.add_argument(
        "--encoder_output",
        type=str,
        default="./nntr_siglip2_encoder_fp32.bin",
        help="Output path for encoder weight file",
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
    """Convert the SigLIP2 encoder to an nntrainer weight file."""
    args = parse_args()

    print(f"Loading checkpoint: {args.model_path}")
    sd = load_file(args.model_path)
    print(f"Checkpoint loaded — {len(sd)} tensors total.")

    dtype = args.data_type

    enc_weights = collect_encoder(sd, dtype)
    print(f"\nEncoder tensors: {len(enc_weights)}")
    assert len(enc_weights) == 199, (
        f"ENCODER COUNT MISMATCH: got {len(enc_weights)}, expected 199"
    )

    if args.safetensors:
        out = get_safetensors_output_name(args.encoder_output)
        save_safetensors(enc_weights, out, dtype)
    else:
        save_bin(enc_weights, args.encoder_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
