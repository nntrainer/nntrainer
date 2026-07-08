# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file verify_encoder.py
## @brief PyTorch reference + nntrainer parity check for the SigLIP2 vision
##        encoder. Emits the golden projected encoder output [1,196,256] and the
##        preprocessed pixel tensor, and (with --nntr-npy) compares the nntrainer
##        dump (nntr_encoder_hidden.npy from `nntr_causallm --dump-encoder`).
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel


def preprocess(image_path, size=224):
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # SigLIP normalization
    arr = arr.transpose(2, 0, 1)[None]  # [1,3,224,224]
    return torch.from_numpy(np.ascontiguousarray(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF source checkpoint dir (VisionEncoderDecoder)")
    ap.add_argument("--image", default="sample.png")
    ap.add_argument("--out", default="golden", help="output prefix")
    ap.add_argument("--nntr-npy", help="nntr_encoder_hidden.npy to compare")
    # Float32 activations differ across implementations only by accumulation
    # order (GEMM / softmax / gelu-tanh), so parity is judged by cosine
    # similarity + relative L2, not raw max-abs (which is dominated by a handful
    # of large-magnitude outliers). Defaults are tight enough that any real
    # weight/graph error fails immediately.
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--max-rel-l2", type=float, default=1e-2)
    args = ap.parse_args()

    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt, attn_implementation="eager"
    ).eval()
    pixel = preprocess(args.image)

    with torch.no_grad():
        enc = model.get_encoder()(pixel_values=pixel, return_dict=True).last_hidden_state
        proj = (
            model.enc_to_dec_proj(enc)
            if getattr(model, "enc_to_dec_proj", None)
            else enc
        )
    golden = proj.cpu().numpy().astype(np.float32)  # [1,196,256]

    np.save(args.out + ".encoder_hidden.npy", golden)
    np.save(args.out + ".pixel.npy", pixel.cpu().numpy())
    print("golden encoder_hidden shape:", golden.shape)
    print("saved:", args.out + ".encoder_hidden.npy,", args.out + ".pixel.npy")

    if args.nntr_npy:
        nntr = np.load(args.nntr_npy).astype(np.float64).reshape(-1)
        ref = golden.astype(np.float64).reshape(-1)
        diff = np.abs(ref - nntr)
        cos = float(ref @ nntr / (np.linalg.norm(ref) * np.linalg.norm(nntr)))
        rel_l2 = float(np.linalg.norm(ref - nntr) / np.linalg.norm(ref))
        print(f"max |diff| = {diff.max():.6e}  mean |diff| = {diff.mean():.6e}")
        print(f"cosine = {cos:.8f} (min {args.min_cos})  "
              f"rel L2 = {rel_l2:.6e} (max {args.max_rel_l2})")
        ok = (cos >= args.min_cos) and (rel_l2 <= args.max_rel_l2)
        print("ENCODER PARITY:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
