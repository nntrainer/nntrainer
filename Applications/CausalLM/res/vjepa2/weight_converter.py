## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
##
## @file weight_converter.py
## @brief Weight conversion for V-JEPA 2.1 ViT-B/16 (video) encoder.
## @author Jijoong Moon <jijoong.moon@samsung.com>
##
## Converts the released V-JEPA 2.1 checkpoint (vjepa2_1_vit_base_384) encoder
## into the nntrainer binary format expected by the VJEPA2ViT model graph.
##
## The byte order of the output exactly follows the order in which VJEPA2ViT
## creates its weight-bearing layers (see models/vjepa2_vit/vjepa2_vit.cpp):
##
##   patch_embed/proj          (FC: weight, bias)
##   for each of 12 blocks:
##     layer{i}_attention_norm (LN: weight, bias)   <- blocks.{i}.norm1
##     layer{i}_qkv_q          (FC: weight, bias)   <- blocks.{i}.attn.qkv[0:768]
##     layer{i}_qkv_k          (FC: weight, bias)   <- blocks.{i}.attn.qkv[768:1536]
##     layer{i}_qkv_v          (FC: weight, bias)   <- blocks.{i}.attn.qkv[1536:2304]
##     layer{i}_attention_out  (FC: weight, bias)   <- blocks.{i}.attn.proj
##     layer{i}_ffn_norm       (LN: weight, bias)   <- blocks.{i}.norm2
##     layer{i}_ffn_up         (FC: weight, bias)   <- blocks.{i}.mlp.fc1
##     layer{i}_ffn_down       (FC: weight, bias)   <- blocks.{i}.mlp.fc2
##   output_norm               (LN: weight, bias)   <- norms_block.{last}
##
## Notes:
##   - The 3D tubelet patch embedding (Conv3d) is non-overlapping, hence exactly
##     equivalent to a Linear over the flattened tubelet. The Conv3d weight
##     [embed, in_ch, kT, kH, kW] is reshaped to [embed, in_ch*kT*kH*kW] (C
##     order, matching VJEPA2ViT::patchify's (c, kt, kh, kw) layout) and then
##     transposed to nntrainer's [in, out] FC layout.
##   - The (video) modality embedding is a constant added to every token, so it
##     is folded into the patch-embed bias here. The image modality path
##     (patch_embed_img / img_mod_embed) is intentionally not exported.
##   - 3D RoPE carries no weights.

import argparse
import numpy as np
import torch


def load_encoder_state(model_path, checkpoint_key):
    """Load the encoder sub state-dict and strip module/backbone prefixes."""
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and checkpoint_key in sd:
        sd = sd[checkpoint_key]
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v
    return cleaned


def save_weight(weight, dtype, file, transpose=False):
    """Save a tensor to nntrainer format (optionally transposing OI -> IO)."""
    array = weight if isinstance(weight, np.ndarray) else weight.detach().cpu().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    array.astype(dtype).tofile(file)


def convert(model_path, output_path, dtype, cfg):
    dim = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    checkpoint_key = cfg["checkpoint_key"]
    final_norm_key = cfg["final_norm_key"]

    print(f"Loading encoder ('{checkpoint_key}') from: {model_path}")
    sd = load_encoder_state(model_path, checkpoint_key)

    # Sanity: surface a few keys if expected ones are missing.
    if "patch_embed.proj.weight" not in sd:
        print("  [warn] 'patch_embed.proj.weight' not found. Available keys:")
        for k in list(sd.keys())[:20]:
            print("    ", k, tuple(sd[k].shape))

    print(f"Writing nntrainer weights to: {output_path}")
    with open(output_path, "wb") as f:
        # 1. Patch embedding (Conv3d -> FC), modality embed folded into bias.
        pw = sd["patch_embed.proj.weight"]            # [embed, in_ch, kT, kH, kW]
        pw = pw.reshape(pw.shape[0], -1)              # [embed, in_ch*kT*kH*kW]
        save_weight(pw, dtype, f, transpose=True)     # -> [in, out]

        pb = sd["patch_embed.proj.bias"].clone()      # [embed]
        if "video_mod_embed" in sd:
            pb = pb + sd["video_mod_embed"].reshape(-1)
            print("  folded video_mod_embed into patch-embed bias")
        save_weight(pb, dtype, f)

        # 2. Transformer blocks.
        for i in range(num_layers):
            p = f"blocks.{i}."

            # norm1 -> attention_norm
            save_weight(sd[p + "norm1.weight"], dtype, f)
            save_weight(sd[p + "norm1.bias"], dtype, f)

            # fused qkv -> q, k, v (each transposed + bias)
            qkv_w = sd[p + "attn.qkv.weight"]          # [3*dim, dim]
            qkv_b = sd[p + "attn.qkv.bias"]            # [3*dim]
            for s in range(3):
                save_weight(qkv_w[s * dim:(s + 1) * dim, :], dtype, f, transpose=True)
                save_weight(qkv_b[s * dim:(s + 1) * dim], dtype, f)

            # attn.proj -> attention_out
            save_weight(sd[p + "attn.proj.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "attn.proj.bias"], dtype, f)

            # norm2 -> ffn_norm
            save_weight(sd[p + "norm2.weight"], dtype, f)
            save_weight(sd[p + "norm2.bias"], dtype, f)

            # mlp.fc1 -> ffn_up, mlp.fc2 -> ffn_down
            save_weight(sd[p + "mlp.fc1.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "mlp.fc1.bias"], dtype, f)
            save_weight(sd[p + "mlp.fc2.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "mlp.fc2.bias"], dtype, f)

            print(f"  layer {i + 1}/{num_layers} done")

        # 3. Final norm (norms_block[-1] in the reference forward).
        save_weight(sd[final_norm_key + ".weight"], dtype, f)
        save_weight(sd[final_norm_key + ".bias"], dtype, f)

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vjepa2_1_vitb_dist_vitG_384.pt",
                        help="Input V-JEPA 2.1 checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="nntr_vjepa2_vitb_fp32.bin",
                        help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--checkpoint_key", type=str, default="ema_encoder",
                        help="Top-level state_dict key holding the encoder")
    parser.add_argument("--final_norm_key", type=str, default="norms_block.3",
                        help="Key of the final LayerNorm (norms_block[-1])")
    args = parser.parse_args()

    dtype = {"float32": np.float32, "float16": np.float16}[args.dtype]
    cfg = dict(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_key=args.checkpoint_key,
        final_norm_key=args.final_norm_key,
    )
    convert(args.input, args.output, dtype, cfg)
