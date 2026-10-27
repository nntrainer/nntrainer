#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   make_reference.py
# @brief  Build a YOLOv7ReIDtiny model (random weights unless --weights given),
#         run the fused PyTorch forward on a fixed input, and dump:
#           <out>/pose_merged_random.pt   (checkpoint, when random)
#           <out>/input_320.bin           (NCHW FP32 input, 1x3x320x320)
#           <out>/ref_pose.bin            (FP32 [2*NKPT, SIMCC_BINS])
#           <out>/ref_reid.bin            (FP32 [EMBED_DIM])
#         Used by verify_parity.py to check the nntrainer port end-to-end.

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.fuse import IFuse  # noqa: E402
from models_pose.pose_ref import YOLOv7Posetiny  # noqa: E402
from weight_converter import _load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="optional real .pt")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", default="", help="optional input image")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = YOLOv7Posetiny(nkpt=87, img_size=320)
    if args.weights:
        state = _load_checkpoint(args.weights, torch.device("cpu"))
        state = {k: v for k, v in state.items() if not k.startswith("model.")}
        already_fused = not any(".bn." in k for k in state)
        if already_fused:
            for mod in model.modules():
                if isinstance(mod, IFuse):
                    mod.fuse()
        miss, unexp = model.load_state_dict(state, strict=False)
        miss = [k for k in miss if not k.startswith("model.")]
        unexp = [k for k in unexp if not k.startswith("model.")]
        if miss or unexp:
            print("missing:", miss[:8])
            print("unexpected:", unexp[:8])
            raise SystemExit("state_dict mismatch")
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
        torch.save(model.state_dict(), os.path.join(args.out, "pose_random.pt"))

    model.eval()
    for mod in model.modules():
        if isinstance(mod, IFuse):
            mod.fuse()

    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB").resize((320, 320))
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))[None]
    else:
        x = torch.randn(1, 3, 320, 320)

    with torch.no_grad():
        pose = model(x)

    x.numpy().astype(np.float32).tofile(os.path.join(args.out, "input_320.bin"))
    pose[0].numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_pose.bin"))
    print("pose", tuple(pose.shape))
    print("wrote input_320.bin, ref_pose.bin to", args.out)


if __name__ == "__main__":
    main()
