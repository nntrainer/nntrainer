#!/usr/bin/env python3
## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
"""Extract CED-tiny reference tensors for the nntrainer port.

Model input boundary is `input_values` (the mel-dB spectrogram), which is what
HF's CedModel.forward() itself takes -- feature extraction lives outside the
model in both implementations.

Writes, under --out-dir:
  input_values.bin        [n_mels, T]        fp32   model input
  ref_init_bn.bin         [n_mels, T]        fp32   after folded init_bn
  ref_patch_pos.bin       [D, F, Tg]        fp32   after patch conv + pos embed
  ref_hidden.bin          [F*Tg, D]         fp32   encoder output (after norm)
  ref_pooled.bin          [D]               fp32   mean over tokens
  ref_logits_presigmoid.bin [n_classes]     fp32
  ref_logits.bin          [n_classes]       fp32   final sigmoid output
  meta.json                                        shapes + top-5 labels
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/seungbaek/hdd/models/ced-tiny")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wav", default="", help="optional real wav (16 kHz)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_dir, trust_remote_code=True)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(args.model_dir,
                                             trust_remote_code=True)
    cfg = model.config
    target_length = cfg.target_length          # 1012
    n_mels = cfg.n_mels                        # 64
    D = cfg.embed_dim                          # 192
    patch = cfg.patch_stride                   # 16

    # ---- input ------------------------------------------------------------
    if args.wav:
        import soundfile as sf
        wav, sr = sf.read(args.wav, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, f"expected 16 kHz, got {sr}"
    else:
        # Exactly target_length frames: center=True gives len//hop + 1 frames.
        n_samples = (target_length - 1) * cfg.hop_size
        rng = np.random.default_rng(args.seed)
        wav = rng.standard_normal(n_samples).astype(np.float32) * 0.1

    feats = fe(wav, sampling_rate=16000).input_values  # [1, n_mels, T]
    if feats.shape[-1] != target_length:
        raise SystemExit(
            f"got T={feats.shape[-1]}, need exactly {target_length} so the "
            "reference stays in the single-split path (no pad/split logic)")
    x = feats.float()
    print(f"[input] input_values {tuple(x.shape)} "
          f"min={x.min():.6f} max={x.max():.6f} mean={x.mean():.6f}")
    x[0].numpy().astype(np.float32).tofile(
        os.path.join(args.out_dir, "input_values.bin"))

    enc = model.encoder
    bn = enc.init_bn
    # ---- folded init_bn (per-freq affine) ---------------------------------
    # BatchNorm2d(n_mels) is applied with freq as the channel axis, so in eval
    # mode it is exactly y[f,t] = a[f]*x[f,t] + b[f].
    a = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    b = (bn.bias - bn.running_mean * a).detach()
    np.stack([a.numpy(), b.numpy()]).astype(np.float32).tofile(
        os.path.join(args.out_dir, "init_bn_affine.bin"))

    with torch.no_grad():
        # reference init_bn exactly as the model runs it
        xb = torch.unsqueeze(x, 1)                     # [1,1,F,T]
        xb = torch.permute(xb, (0, 2, 1, 3))           # [1,F,1,T]
        xb = bn(xb)
        xb = torch.permute(xb, (0, 2, 1, 3))           # [1,1,F,T]
        folded = a[None, None, :, None] * torch.unsqueeze(x, 1) \
            + b[None, None, :, None]
        fold_diff = (xb - folded).abs().max().item()
        print(f"[init_bn] fold max_abs_diff vs nn.BatchNorm2d = {fold_diff:.3e}")
        xb[0, 0].numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_init_bn.bin"))

        # patch embed + positional embeddings, still in [B,D,F,Tg]
        p = enc.patch_embed(xb)
        grid_f, grid_t = p.shape[2], p.shape[3]
        p = p + enc.time_pos_embed[:, :, :, :grid_t]
        p = p + enc.freq_pos_embed[:, :, :, :]
        print(f"[patch] {tuple(p.shape)} grid=({grid_f},{grid_t}) "
              f"tokens={grid_f * grid_t}")
        p[0].numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_patch_pos.bin"))

        # combined pos-embed table, flattened exactly like the model does
        # (flatten(2,3) is freq-major then time)
        pos = (enc.time_pos_embed[:, :, :, :grid_t]
               + enc.freq_pos_embed[:, :, :, :])       # [1,D,F,Tg] broadcast
        pos = torch.permute(torch.flatten(pos, 2, 3), (0, 2, 1))  # [1,N,D]
        pos[0].numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "pos_embed_combined.bin"))

        hidden = enc(x).logits                          # [1, N, D]
        hidden = hidden.reshape(-1, D)
        print(f"[hidden] {tuple(hidden.shape)} "
              f"min={hidden.min():.6f} max={hidden.max():.6f}")
        hidden.numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_hidden.bin"))

        pooled = hidden.mean(0)
        pooled.numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_pooled.bin"))

        pre = model.outputlayer(pooled)
        pre.numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_logits_presigmoid.bin"))

        out = model(x).logits[0]                        # sigmoid applied
        print(f"[logits] {tuple(out.shape)} "
              f"min={out.min():.6f} max={out.max():.6f}")
        chk = (out - pre.sigmoid()).abs().max().item()
        print(f"[logits] path-consistency max_abs_diff = {chk:.3e}")
        out.numpy().astype(np.float32).tofile(
            os.path.join(args.out_dir, "ref_logits.bin"))

    top = torch.topk(out, 5)
    def label_of(idx):
        m = cfg.id2label
        return m.get(idx, m.get(str(idx), f"<{idx}>"))

    labels = [(label_of(int(i)), round(float(v), 6))
              for v, i in zip(top.values, top.indices)]
    print(f"[top5] {labels}")

    meta = {
        "n_mels": n_mels, "target_length": target_length, "embed_dim": D,
        "patch": patch, "grid": [grid_f, grid_t],
        "num_patches": grid_f * grid_t, "depth": cfg.depth,
        "num_heads": cfg.num_heads, "outputdim": cfg.outputdim,
        "mlp_hidden": int(D * cfg.mlp_ratio), "norm_eps": 1e-6,
        "bn_eps": bn.eps, "init_bn_fold_max_diff": fold_diff,
        "top5": labels,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
