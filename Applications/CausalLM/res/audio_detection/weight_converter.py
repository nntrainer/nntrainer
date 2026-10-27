#!/usr/bin/env python3
## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight + config conversion for the audio-detection model
##        (ced_tiny backbone, xi-vector pooling, 12-class SED head)
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## Emits an nntrainer model directory: config.json, nntr_config.json and a flat
## FP32 weight blob ordered to match CedTransformer's layer creation order:
##
##   input_norm        moving_mean, moving_variance, gamma, beta   (init_bn)
##   patch_embed/conv  weight [D,1,P,P], bias [D]
##   pos_embed/weights combined time+freq table [1,1,N,D], sliced to the grid
##   layer{i}_...      norm1, q, k, v, proj, norm2, fc1, fc2       (x depth)
##   output_norm       weight, bias
##   head/pool         xi: lin1 w/b, bn mean/var/gamma/beta, lin2 w/b,
##                     prior_mean, prior_logprec
##   head/norm         weight, bias                        (outputlayer.0)
##   head/classifier   weight [D,C], bias [C]              (outputlayer.1)

import argparse
import json
import os

import numpy as np
import torch

LABEL_NAMES = [
    "dog_bark", "cat_meow", "doorbell", "glass_break", "clap", "knock",
    "alert", "speech", "baby_cry", "cough", "scream", "water",
]
# Full precision, as they appear in the deployed label yaml. Rounding these to
# three decimals shifts a threshold by up to 5e-4, which is the same order as
# the score differences between runtimes, so it can flip a borderline decision.
THRESHOLDS = {
    "dog_bark": 0.6420, "cat_meow": 0.8111, "doorbell": 0.9407,
    "glass_break": 0.8465, "clap": 0.9820, "knock": 0.9851, "alert": 0.6942,
    "speech": 0.4802, "baby_cry": 0.9257, "cough": 0.8951, "scream": 0.9531,
    "water": 0.8807,
}


def save(t, f, transpose=False):
    """Append one tensor to the blob, optionally [out,in] -> [in,out]."""
    a = t if isinstance(t, np.ndarray) else t.detach().cpu().numpy()
    if transpose and a.ndim >= 2:
        a = a.T
    a.astype(np.float32).tofile(f)


def combined_pos_embed(sd, prefix, grid_t):
    """Pre-sum the separable time/freq positional tables into one flat table.

    The model adds them in [B, D, F, T] layout before flattening, and only uses
    the first `grid_t` columns of the time table (the table is sized for the
    pretraining window, which is longer than the deployed one). Both are
    constants, so the broadcast sum folds exactly into the single
    [1, 1, F*T, D] table the nntrainer graph adds after flattening.
    """
    time_pos = sd[f"{prefix}time_pos_embed"][:, :, :, :grid_t]
    freq_pos = sd[f"{prefix}freq_pos_embed"]
    pos = time_pos + freq_pos
    pos = torch.permute(torch.flatten(pos, 2, 3), (0, 2, 1))
    return pos.unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint directory holding weights.pt/config.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=101,
                    help="mel frames per inference window "
                         "(1 s at hop 160 with center padding = 101)")
    ap.add_argument("--window-samples", type=int, default=16000)
    ap.add_argument("--stride-samples", type=int, default=16000)
    ap.add_argument("--log-mode", choices=["db", "natural"], default="db",
                    help="'db' = 10log10 + top_db (torchaudio AmplitudeToDB); "
                         "'natural' = ln(max(mel, log_eps))")
    ap.add_argument("--log-eps", type=float, default=1e-5)
    ap.add_argument("--normalize-divisor", type=float, default=32768.0,
                    help="int16 -> float divisor; 32767 matches pipelines that "
                         "use Java's Short.MAX_VALUE")
    ap.add_argument("--bin-name", default="nntr_audio_detection_fp32.bin")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sd = torch.load(os.path.join(args.ckpt, "weights.pt"), map_location="cpu",
                    weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    p = "transformer."

    dim = sd[p + "norm.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[2]) for k in sd
                    if k.startswith(p + "blocks."))
    n_mels = sd[p + "init_bn.1.weight"].shape[0]
    n_classes = sd[p + "outputlayer.1.weight"].shape[0]
    hidden = sd[p + "pooling_layer.lin1_relu_bn.0.weight"].shape[0]
    patch = sd[p + "patch_embed.proj.weight"].shape[-1]
    heads = {192: 3, 384: 6, 768: 12}.get(dim, dim // 64)
    mlp_hidden = sd[p + "blocks.0.mlp.fc1.weight"].shape[0]

    grid_f = n_mels // patch
    grid_t = args.frames // patch
    tokens = grid_f * grid_t
    print(f"[model] dim={dim} depth={depth} heads={heads} "
          f"mlp_hidden={mlp_hidden} n_mels={n_mels} classes={n_classes}")
    print(f"[shape] frames={args.frames} grid=({grid_f},{grid_t}) "
          f"tokens={tokens} xi_hidden={hidden}")

    bin_path = os.path.join(args.out_dir, args.bin_name)
    with open(bin_path, "wb") as f:
        # 1. init_bn -> the graph's `input_norm` layer.
        save(sd[p + "init_bn.1.running_mean"], f)
        save(sd[p + "init_bn.1.running_var"], f)
        save(sd[p + "init_bn.1.weight"], f)
        save(sd[p + "init_bn.1.bias"], f)

        # 2. patch conv: [out, in, kh, kw] needs no transpose.
        save(sd[p + "patch_embed.proj.weight"], f)
        save(sd[p + "patch_embed.proj.bias"], f)

        # 3. positional table.
        save(combined_pos_embed(sd, p, grid_t), f)

        # 4. encoder blocks.
        for i in range(depth):
            b = f"{p}blocks.{i}."
            save(sd[b + "norm1.weight"], f)
            save(sd[b + "norm1.bias"], f)
            qkv_w, qkv_b = sd[b + "attn.qkv.weight"], sd[b + "attn.qkv.bias"]
            for s in range(3):
                save(qkv_w[s * dim:(s + 1) * dim, :], f, transpose=True)
                save(qkv_b[s * dim:(s + 1) * dim], f)
            save(sd[b + "attn.proj.weight"], f, transpose=True)
            save(sd[b + "attn.proj.bias"], f)
            save(sd[b + "norm2.weight"], f)
            save(sd[b + "norm2.bias"], f)
            save(sd[b + "mlp.fc1.weight"], f, transpose=True)
            save(sd[b + "mlp.fc1.bias"], f)
            save(sd[b + "mlp.fc2.weight"], f, transpose=True)
            save(sd[b + "mlp.fc2.bias"], f)

        # 5. encoder output norm.
        save(sd[p + "norm.weight"], f)
        save(sd[p + "norm.bias"], f)

        # 6. xi pooling. The Conv1d kernels are k=1, so [out, in, 1] squeezes to
        # [out, in], which is the row-major layout XiPoolingLayer indexes.
        q = p + "pooling_layer."
        save(sd[q + "lin1_relu_bn.0.weight"].squeeze(-1), f)
        save(sd[q + "lin1_relu_bn.0.bias"], f)
        save(sd[q + "lin1_relu_bn.2.running_mean"], f)
        save(sd[q + "lin1_relu_bn.2.running_var"], f)
        save(sd[q + "lin1_relu_bn.2.weight"], f)
        save(sd[q + "lin1_relu_bn.2.bias"], f)
        save(sd[q + "lin2.weight"].squeeze(-1), f)
        save(sd[q + "lin2.bias"], f)
        save(sd[q + "prior_mean"], f)
        save(sd[q + "prior_logprec"], f)

        # 7. head.
        save(sd[p + "outputlayer.0.weight"], f)
        save(sd[p + "outputlayer.0.bias"], f)
        save(sd[p + "outputlayer.1.weight"], f, transpose=True)
        save(sd[p + "outputlayer.1.bias"], f)

    print(f"[bin]  {bin_path} ({os.path.getsize(bin_path)} bytes)")

    cfg = {
        "architectures": ["CedForAudioClassification"],
        "model_type": "ced",
        "name": "audio-detection-ced-tiny",
        "embed_dim": dim,
        "depth": depth,
        "num_heads": heads,
        "mlp_ratio": mlp_hidden / dim,
        "qkv_bias": True,
        "patch_size": patch,
        "patch_stride": patch,
        "n_mels": n_mels,
        # Input width is the mel frame count of one inference window, not the
        # pretraining target_length; the positional table is sliced to match.
        "target_length": args.frames,
        "outputdim": n_classes,
        "pooling": "xi",
        "pooling_hidden_size": hidden,
        "id2label": {str(i): l for i, l in enumerate(LABEL_NAMES)},
        "label2id": {l: i for i, l in enumerate(LABEL_NAMES)},
        "front_end": {
            "sample_rate": 16000, "n_fft": 512, "win_size": 512,
            "hop_size": 160, "center": True, "f_min": 0, "f_max": 8000,
            "top_db": 80.0, "power": 2.0,
            "log_mode": args.log_mode, "log_eps": args.log_eps,
            "window_samples": args.window_samples,
            "stride_samples": args.stride_samples,
            "normalize_divisor": args.normalize_divisor,
        },
        "thresholds": THRESHOLDS,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    nntr_cfg = {
        "model_tensor_type": "FP32-FP32",
        "model_file_name": args.bin_name,
        "model_type": "Model",
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "head_dtype": "FP32",
        "batch_size": 1,
        "sample_input": "dog_tizen.wav",
        "init_seq_len": tokens,
        "max_seq_len": tokens,
        "num_to_generate": 0,
        "fsu": False,
        "skip_tokenizer": True,
    }
    with open(os.path.join(args.out_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_cfg, f, indent=2)

    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
