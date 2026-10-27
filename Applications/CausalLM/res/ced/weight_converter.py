## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for the CED audio tagging model
##        (https://huggingface.co/mispeech/ced-tiny)
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## The output is a flat FP32 blob whose order matches the layer creation order
## of CedTransformer / TimmViTTransformer:
##
##   input_norm          moving_mean, moving_variance, gamma, beta   (init_bn)
##   patch_embed/conv    weight [D,1,P,P], bias [D]
##   pos_embed/weights   combined time+freq table [1,1,N,D]
##   layer{i}_...        norm1 w/b, q w/b, k w/b, v w/b, proj w/b,
##                       norm2 w/b, fc1 w/b, fc2 w/b                 (x depth)
##   output_norm         w, b
##   head/norm           w, b                                (outputlayer.0)
##   head/classifier     w [D,C], b [C]                       (outputlayer.1)

import argparse
import json
import os

import numpy as np
import safetensors.torch
import torch


def load_state_dict(model_path):
    """Load the CED checkpoint from safetensors or a PyTorch bin."""
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.safetensors")

    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
        for key in ("state_dict", "model"):
            if isinstance(state_dict, dict) and isinstance(
                    state_dict.get(key), dict):
                state_dict = state_dict[key]
                break

    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def save_weight(weight, dtype, file, transpose=False):
    """Append one tensor to the nntrainer weight blob.

    transpose=True converts PyTorch's [out, in] Linear layout to the [in, out]
    layout nntrainer's fully_connected expects.
    """
    array = weight if isinstance(weight, np.ndarray) \
        else weight.detach().cpu().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    array.astype(dtype).tofile(file)


def combined_pos_embed(state_dict, grid_t):
    """Pre-sum CED's two separable positional embedding tables.

    modeling_ced.py adds them in [B, D, F, T] layout and only then flattens:

        x = x + time_pos_embed[:, :, :, :t]     # [1, D, 1, T]
        x = x + freq_pos_embed                  # [1, D, F, 1]
        x = permute(flatten(x, 2, 3), (0, 2, 1))

    Both are constants, so their broadcast sum can be folded into the single
    [1, 1, F*T, D] table the nntrainer graph adds after flattening. flatten(2,3)
    is frequency-major, which is also how a [D, F, T] NCHW tensor reshapes to
    [D, F*T], so the two orderings agree.
    """
    time_pos = state_dict["encoder.time_pos_embed"][:, :, :, :grid_t]
    freq_pos = state_dict["encoder.freq_pos_embed"]
    pos = time_pos + freq_pos                       # [1, D, F, T]
    pos = torch.permute(torch.flatten(pos, 2, 3), (0, 2, 1))  # [1, F*T, D]
    return pos.unsqueeze(1)                         # [1, 1, F*T, D]


def convert_ced_to_nntrainer(model_path, output_path, config_path=None,
                            dtype=np.float32):
    """Convert CED weights to the nntrainer flat weight format."""
    print(f"Loading model from: {model_path}")
    state_dict = load_state_dict(model_path)

    if config_path is None:
        base = model_path if os.path.isdir(model_path) \
            else os.path.dirname(model_path)
        config_path = os.path.join(base, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    dim = cfg["embed_dim"]
    depth = cfg["depth"]
    n_mels = cfg["n_mels"]
    target_length = cfg["target_length"]
    patch = cfg["patch_size"]
    stride = cfg.get("patch_stride", patch)
    grid_f = n_mels // stride
    grid_t = target_length // stride

    print(f"  embed_dim={dim} depth={depth} grid=({grid_f},{grid_t}) "
          f"tokens={grid_f * grid_t} classes={cfg['outputdim']}")
    print(f"Converting weights to: {output_path}")

    with open(output_path, "wb") as f:
        # 1. init_bn -> the graph's `input_norm` batch_normalization layer.
        # nntrainer registers BN weights as moving_mean, moving_variance,
        # gamma, beta -- in that order.
        print("  Processing init_bn...")
        save_weight(state_dict["encoder.init_bn.running_mean"], dtype, f)
        save_weight(state_dict["encoder.init_bn.running_var"], dtype, f)
        save_weight(state_dict["encoder.init_bn.weight"], dtype, f)
        save_weight(state_dict["encoder.init_bn.bias"], dtype, f)

        # 2. Patch embedding conv: [out, in, kh, kw] needs no transpose.
        print("  Processing patch embedding...")
        save_weight(state_dict["encoder.patch_embed.proj.weight"], dtype, f)
        save_weight(state_dict["encoder.patch_embed.proj.bias"], dtype, f)

        # 3. Combined positional embedding.
        print("  Processing positional embeddings (time + freq)...")
        save_weight(combined_pos_embed(state_dict, grid_t), dtype, f)

        # 4. Encoder blocks.
        print(f"  Processing {depth} transformer blocks...")
        for i in range(depth):
            p = f"encoder.blocks.{i}."

            save_weight(state_dict[p + "norm1.weight"], dtype, f)
            save_weight(state_dict[p + "norm1.bias"], dtype, f)

            # Fused qkv split into the graph's separate q -> k -> v layers.
            qkv_w = state_dict[p + "attn.qkv.weight"]
            qkv_b = state_dict[p + "attn.qkv.bias"]
            for s in range(3):
                save_weight(qkv_w[s * dim:(s + 1) * dim, :], dtype, f,
                            transpose=True)
                save_weight(qkv_b[s * dim:(s + 1) * dim], dtype, f)

            save_weight(state_dict[p + "attn.proj.weight"], dtype, f,
                        transpose=True)
            save_weight(state_dict[p + "attn.proj.bias"], dtype, f)

            save_weight(state_dict[p + "norm2.weight"], dtype, f)
            save_weight(state_dict[p + "norm2.bias"], dtype, f)

            save_weight(state_dict[p + "mlp.fc1.weight"], dtype, f,
                        transpose=True)
            save_weight(state_dict[p + "mlp.fc1.bias"], dtype, f)
            save_weight(state_dict[p + "mlp.fc2.weight"], dtype, f,
                        transpose=True)
            save_weight(state_dict[p + "mlp.fc2.bias"], dtype, f)

            print(f"    Layer {i + 1}/{depth} done")

        # 5. Encoder output LayerNorm.
        print("  Processing final normalization...")
        save_weight(state_dict["encoder.norm.weight"], dtype, f)
        save_weight(state_dict["encoder.norm.bias"], dtype, f)

        # 6. Head: outputlayer = Sequential(LayerNorm, Linear).
        print("  Processing classification head...")
        save_weight(state_dict["outputlayer.0.weight"], dtype, f)
        save_weight(state_dict["outputlayer.0.bias"], dtype, f)
        save_weight(state_dict["outputlayer.1.weight"], dtype, f,
                    transpose=True)
        save_weight(state_dict["outputlayer.1.bias"], dtype, f)

    print("\nConversion complete!")
    print(f"  {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="model.safetensors",
                        help="CED checkpoint directory or .safetensors/.bin")
    parser.add_argument("--output", type=str, default="nntr_ced_fp32.bin",
                        help="output nntrainer weight file")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json (default: alongside the input)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    args = parser.parse_args()

    convert_ced_to_nntrainer(
        args.input, args.output, args.config,
        {"float32": np.float32, "float16": np.float16}[args.dtype])
