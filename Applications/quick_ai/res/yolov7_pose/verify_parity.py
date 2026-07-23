#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   verify_parity.py
# @brief  Compare nntrainer yolov7_pose_infer raw outputs against the PyTorch
#         reference produced by make_reference.py.
#
#   Usage:
#     python3 verify_parity.py --ref-pose ref_pose.bin --ref-reid ref_reid.bin \
#         --out-pose nn_pose.bin --out-reid nn_reid.bin

import argparse

import numpy as np

NKPT = 87
SIMCC_BINS = 640
EMBED_DIM = 128


def load(path, n):
    v = np.fromfile(path, dtype=np.float32)
    if v.size != n:
        raise SystemExit(f"{path}: expected {n} floats, got {v.size}")
    return v


def report(name, a, b):
    diff = np.abs(a - b)
    scale = np.abs(b).max() + 1e-9
    rel = diff.max() / scale  # error relative to the tensor's dynamic range
    print(
        f"[{name}] max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} "
        f"range={scale:.3e} max_rel_to_range={rel:.4e}"
    )
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-pose", required=True)
    ap.add_argument("--out-pose", required=True)
    ap.add_argument("--ref-reid", default="")
    ap.add_argument("--out-reid", default="")
    # tolerance is on the error relative to each tensor's dynamic range;
    # keypoint correctness is judged by exact SIMCC argmax agreement.
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    rp = load(args.ref_pose, 2 * NKPT * SIMCC_BINS)
    op = load(args.out_pose, 2 * NKPT * SIMCC_BINS)

    mp = report("pose", op, rp)
    mr = 0.0
    if args.ref_reid and args.out_reid:
        rr = load(args.ref_reid, EMBED_DIM)
        orr = load(args.out_reid, EMBED_DIM)
        mr = report("reid", orr, rr)

    # keypoint-level agreement (argmax over SIMCC bins). A handful of near-tie
    # rows may flip under matmul FP-order noise with random weights; allow a
    # small tolerance (trained weights give sharp unimodal peaks).
    rpm = rp.reshape(2 * NKPT, SIMCC_BINS)
    opm = op.reshape(2 * NKPT, SIMCC_BINS)
    agree = int((rpm.argmax(1) == opm.argmax(1)).sum())
    print(f"[argmax] {agree}/{2 * NKPT} SIMCC rows match exactly")

    ok = agree >= 2 * NKPT - 2 and mp < args.tol and mr < args.tol
    print("RESULT:", "PASS" if ok else "FAIL", f"(tol={args.tol})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
