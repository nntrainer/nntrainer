#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   make_input.py
# @brief  Turn an RGB image into the NCHW FP32 input tensor the pose app reads
#         (matches the reference inference.py preprocessing: resize 320x320,
#         /255, channel-first). Writes <out>/input_320.bin (1x3x320x320).
#
#   python3 make_input.py --image sample.jpg --out .

import argparse

import numpy as np
from PIL import Image

INPUT_SIZE = 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0        # HWC
    chw = arr.transpose(2, 0, 1)[None]                       # 1,C,H,W
    path = f"{args.out}/input_320.bin"
    chw.astype(np.float32).tofile(path)
    print(f"wrote {path}  shape={chw.shape}")


if __name__ == "__main__":
    main()
