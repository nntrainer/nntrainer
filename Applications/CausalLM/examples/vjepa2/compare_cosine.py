#!/usr/bin/env python3
## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
##
## @file compare_cosine.py
## @brief Compare an nntrainer VJEPA2ViT dump against the torch reference.
## @author Jijoong Moon <jijoong.moon@samsung.com>
##
"""Compare an nntrainer VJEPA2ViT dump against the torch reference output.

The app dumps the **token-0** hidden state (DIM=768 float32) to
``<input>.nntr_out.bin``.  The torch reference ``ref_output.npy`` is the full
``[num_patches, dim]`` encoder output; we compare against row 0.

Usage:
  compare_cosine.py --ref ref_output.npy --nntr input_video.bin.nntr_out.bin
  compare_cosine.py --ref ref_output.npy --values "0.0696 0.7793 ..."   # from stdout
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="torch ref_output.npy [N, dim]")
    ap.add_argument("--nntr", help="raw float32 nntrainer dump (token0 or full)")
    ap.add_argument("--values", help="space-separated token-0 values printed by the app")
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    ref = np.load(args.ref)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    print(f"ref shape: {ref.shape}  mean/std: {ref.mean():.5f}/{ref.std():.5f}")

    if args.values:
        nntr = np.array([float(x) for x in args.values.split()], dtype=np.float32)
    elif args.nntr:
        nntr = np.fromfile(args.nntr, dtype=np.float32)
    else:
        ap.error("need --nntr or --values")

    # The app dumps token 0 only (DIM floats); compare against ref[0].
    n = min(nntr.size, ref.shape[1])
    a = nntr.reshape(-1)[:n].astype(np.float64)
    b = ref[0, :n].astype(np.float64)

    diff = np.abs(a - b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"compared {n} dims of token 0")
    print(f"  max abs diff : {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  cosine       : {cos:.6f}")
    # Encoder output cosine >= 0.985 is the pass bar for the Q4_0 device path.
    ok = cos >= 0.985
    print("RESULT:", "PASS" if ok else "FAIL", f"(cosine {cos:.4f} vs 0.985 bar)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
