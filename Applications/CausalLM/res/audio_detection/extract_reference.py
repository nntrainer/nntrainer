#!/usr/bin/env python3
## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
"""Extract audio-detection (ced_tiny + xi pooling, 12 classes) reference tensors.

Mirrors run_demo.py exactly, and additionally dumps every intermediate the
nntrainer port needs to bisect against:

  audio.bin            [n_samples]        fp32  int16/32768 normalized mono
  window{i}_mel.bin    [n_mels, frames]   fp32  FrontEnd output (mel dB)
  window{i}_initbn.bin [n_mels, frames]   fp32  after init_bn
  window{i}_hidden.bin [tokens, D]        fp32  after backbone final norm
  window{i}_pooled.bin [D]                fp32  after xi pooling
  window{i}_logits.bin [n_classes]        fp32  outputlayer output (pre-sigmoid)
  window{i}_probs.bin  [n_classes]        fp32  sigmoid(logits)  <- final output
  stft_window.bin      [n_fft]            fp32  Hann window from the checkpoint
  mel_fb.bin           [n_stft, n_mels]   fp32  mel filterbank from the ckpt
  expected.txt                                  the run_demo.py table
  meta.json                                     shapes and label/threshold info
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

LABEL_NAMES = [
    "dog_bark", "cat_meow", "doorbell", "glass_break", "clap", "knock",
    "alert", "speech", "baby_cry", "cough", "scream", "water",
]
THRESHOLDS = {
    "dog_bark": 0.642, "cat_meow": 0.811, "doorbell": 0.941,
    "glass_break": 0.847, "clap": 0.982, "knock": 0.985, "alert": 0.694,
    "speech": 0.48, "baby_cry": 0.926, "cough": 0.895, "scream": 0.953,
    "water": 0.881,
}
SAMPLE_RATE = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/seungbaek/projects/0824/"
                                     "pytorch_audio_detection/pytorch")
    ap.add_argument("--wav", default="dog_tizen.wav")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    os.makedirs(args.out_dir, exist_ok=True)

    from pathlib import Path
    from inference_model import ConfigurableModel

    model = ConfigurableModel(checkpoint_path=Path("ckpt"))
    tr = model.model.transformer
    frontend = tr.front_end
    n_mels = tr.n_mels
    D = tr.embed_dim
    # tr.outputdim is the pretrained 527-class value; the fine-tuned head is
    # what actually decides the output width.
    n_classes = tr.outputlayer[1].out_features
    print(f"[model] embed_dim={D} n_mels={n_mels} classes={n_classes} "
          f"pooling={tr.pooling} pooling_out_dim={tr.pooling_out_dim}")

    # ---- front-end constants straight out of the checkpoint ---------------
    spec = frontend[0].spectrogram
    win = spec.window.detach().cpu().numpy().astype(np.float32)
    fb = frontend[0].mel_scale.fb.detach().cpu().numpy().astype(np.float32)
    win.tofile(os.path.join(args.out_dir, "stft_window.bin"))
    fb.tofile(os.path.join(args.out_dir, "mel_fb.bin"))
    print(f"[frontend] n_fft={spec.n_fft} hop={spec.hop_length} "
          f"win_length={spec.win_length} center={spec.center} "
          f"power={spec.power} pad_mode={spec.pad_mode} "
          f"window{win.shape} fb{fb.shape}")

    # ---- preprocessing, byte-for-byte as run_demo.py does it --------------
    audio_int16, sr = sf.read(args.wav, dtype="int16")
    if audio_int16.ndim > 1:
        audio_int16 = audio_int16[:, 0]
    if sr != SAMPLE_RATE:
        audio_int16 = signal.resample_poly(
            audio_int16.astype(np.float32), SAMPLE_RATE, sr).astype(np.int16)
    audio = audio_int16.astype(np.float32) / 32768.0
    audio.tofile(os.path.join(args.out_dir, "audio.bin"))
    print(f"[audio] {args.wav} sr={sr} samples={audio.shape[0]} "
          f"({audio.shape[0] / SAMPLE_RATE:.2f}s) "
          f"min={audio.min():.6f} max={audio.max():.6f}")

    rows = []
    n_windows = 0
    offset = 0
    grid = None
    with torch.no_grad():
        while offset + SAMPLE_RATE <= len(audio):
            w = torch.from_numpy(audio[offset:offset + SAMPLE_RATE]).float()
            mel = frontend(w.unsqueeze(0))          # [1, n_mels, frames]
            mel_np = mel[0].numpy().astype(np.float32)

            # init_bn, exactly as the model applies it
            x = torch.unsqueeze(mel, 1)             # [1,1,F,T]
            xb = tr.init_bn(x)
            hidden = tr.forward_features(xb)        # [1, tokens, D]
            pooled = tr.get_embedding(hidden)       # [1, D] via xi pooling
            logits = tr.outputlayer(pooled)[0]      # [n_classes]
            probs = torch.sigmoid(logits)

            if grid is None:
                p = tr.patch_embed(xb)
                grid = (int(p.shape[2]), int(p.shape[3]))
                print(f"[shapes] mel{tuple(mel_np.shape)} grid={grid} "
                      f"tokens={hidden.shape[1]} pooled{tuple(pooled.shape)}")

            i = n_windows
            mel_np.tofile(os.path.join(args.out_dir, f"window{i}_mel.bin"))
            xb[0, 0].numpy().astype(np.float32).tofile(
                os.path.join(args.out_dir, f"window{i}_initbn.bin"))
            hidden[0].numpy().astype(np.float32).tofile(
                os.path.join(args.out_dir, f"window{i}_hidden.bin"))
            pooled[0].numpy().astype(np.float32).tofile(
                os.path.join(args.out_dir, f"window{i}_pooled.bin"))
            logits.numpy().astype(np.float32).tofile(
                os.path.join(args.out_dir, f"window{i}_logits.bin"))
            probs.numpy().astype(np.float32).tofile(
                os.path.join(args.out_dir, f"window{i}_probs.bin"))

            p_np = probs.numpy()
            scored = sorted(
                ((LABEL_NAMES[k], float(p_np[k])) for k in range(n_classes)),
                key=lambda kv: -kv[1])
            detected = [lbl for lbl, sc in scored if sc >= THRESHOLDS[lbl]]
            rows.append((n_windows, scored, detected))

            offset += SAMPLE_RATE
            n_windows += 1

    lines = [f"=== {os.path.basename(args.wav)} ({n_windows} windows) ==="]
    for i, scored, detected in rows:
        top3 = ", ".join(f"{l}={s:.3f}" for l, s in scored[:3])
        mark = f"  <-- DETECTED: {detected}" if detected else ""
        lines.append(f"  [{i:>4d}-{i + 1:>4d}s] {top3}{mark}")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.out_dir, "expected.txt"), "w") as f:
        f.write(table + "\n")

    meta = {
        "n_mels": n_mels, "embed_dim": D, "n_classes": n_classes,
        "pooling": tr.pooling, "pooling_out_dim": int(tr.pooling_out_dim),
        "pooling_hidden": int(tr.pooling_layer.lin1_relu_bn[0].out_channels),
        "depth": len(tr.blocks), "num_heads": tr.blocks[0].attn.num_heads,
        "patch_size": 16, "patch_stride": 16,
        "grid": list(grid), "tokens": grid[0] * grid[1],
        "frames": int(mel_np.shape[1]), "n_windows": n_windows,
        "n_fft": int(spec.n_fft), "hop": int(spec.hop_length),
        "win_length": int(spec.win_length), "center": bool(spec.center),
        "power": float(spec.power), "pad_mode": str(spec.pad_mode),
        "top_db": 80.0, "sample_rate": SAMPLE_RATE,
        "window_samples": SAMPLE_RATE, "stride_samples": SAMPLE_RATE,
        "labels": LABEL_NAMES, "thresholds": THRESHOLDS,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
