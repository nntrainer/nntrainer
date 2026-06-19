# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file run_pytorch_caption.py
## @brief Minimal PyTorch reference: greedy-caption an image, print tokens+text
##        (same preprocessing nntrainer uses), to compare against nntrainer output.

import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel

DEFAULT_CKPT = ("/home/leeseunghui/workspace/models/"
                "screenaivttprobe-v2.1.0-s02-pytorch/caption_s02")


def preprocess(image_path, size=224):
    # Must match nntrainer: PIL BILINEAR resize, (x/255 - 0.5) / 0.5, CHW.
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)[None]  # [1,3,224,224]
    return torch.from_numpy(np.ascontiguousarray(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="HF caption_s02 checkpoint dir")
    ap.add_argument("--image", default="sample.png")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", help="run the model in float16")
    args = ap.parse_args()

    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    pixel = preprocess(args.image)
    if args.fp16:
        model = model.half()
        pixel = pixel.half()

    with torch.no_grad():
        gen = model.generate(pixel_values=pixel, max_new_tokens=args.max_new_tokens,
                             num_beams=1, do_sample=False)  # greedy

    ids = gen[0].tolist()
    print("image     :", args.image, "(fp16)" if args.fp16 else "(fp32)")
    print("token_ids :", " ".join(map(str, ids)))
    print("caption   :", tok.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
