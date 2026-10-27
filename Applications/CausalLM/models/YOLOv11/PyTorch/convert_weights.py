#!/usr/bin/env python3
"""
@file convert_weights.py
@brief Convert YOLOv11m ultralytics (.pt) weights into a single nntrainer-compatible
safetensors file, with tensor names matching the nntrainer model's weight names
(as built by Applications/CausalLM/models/YOLOv11/jni/main.cpp). The example
then loads everything with one call: model->load(file, SAFETENSORS).

Inference-only: Conv+BatchNorm pairs are fused (ultralytics model.fuse()) so
each Conv module exports a single biased convolution and no BatchNorm tensors.
This removes the BN layers from the nntrainer graph entirely (fewer ops, lower
memory, and no mixed-precision BN path), with no accuracy change since BN is an
exact affine fold at inference time.

Naming scheme (matches nntrainer Weight::getName()):
  conv2d weight : "{layer}/conv:filter"        shape [out, in, kh, kw]
  conv2d bias   : "{layer}/conv:bias"          shape [1, C, 1, 1]
  depthwise     : "{layer}/dw:filter"          shape [C, 1, kh, kw]
  depthwise bias: "{layer}/dw:bias"            shape [1, C, 1, 1]
(depthwise convolution is a grouped conv2d, groups == channels.)

Usage:
  python convert_weights.py --weights v11m_832rect_best.pt --out ../res/yolov11m.safetensors
"""
import os
import json
import struct
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# nntrainer layer-name for each YOLOv11 module (mirrors main.cpp graph names).
BACKBONE = {
    "model.0": "conv0", "model.1": "conv1", "model.3": "conv3",
    "model.5": "conv5", "model.7": "conv7", "model.17": "m17", "model.20": "m20",
}
C3K2 = {  # PyTorch module -> nntrainer base name
    "model.2": "m2", "model.4": "m4", "model.6": "m6", "model.8": "m8",
    "model.13": "m13", "model.16": "m16", "model.19": "m19", "model.22": "m22",
}


class Pack:
    """Accumulates {name: float32 ndarray} and writes one safetensors file."""

    def __init__(self, sd):
        self.sd = sd
        self.T = {}

    def _np(self, key):
        return self.sd[key].detach().cpu().float().numpy().astype(np.float32)

    def conv_bn(self, pt, nn, dw=False):
        # A PyTorch Conv module (conv+BN[+act]) already fused by model.fuse():
        # the module now holds a single biased convolution and no BN tensors.
        w = self._np(f"{pt}.conv.weight")
        b = self._np(f"{pt}.conv.bias")
        key = "dw" if dw else "conv"
        self.T[f"{nn}/{key}:filter"] = w
        self.T[f"{nn}/{key}:bias"] = b.reshape(1, w.shape[0], 1, 1)

    def conv_bias(self, pt, nn):
        w = self._np(f"{pt}.weight")
        self.T[f"{nn}/conv:filter"] = w
        self.T[f"{nn}/conv:bias"] = self._np(f"{pt}.bias").reshape(1, w.shape[0], 1, 1)

    def c3k2(self, pt, nn):
        self.conv_bn(f"{pt}.cv1", f"{nn}/cv1")
        self.conv_bn(f"{pt}.cv2", f"{nn}/cv2")
        m, o = f"{pt}.m.0", f"{nn}/m0"
        self.conv_bn(f"{m}.cv1", f"{o}/cv1")
        self.conv_bn(f"{m}.cv2", f"{o}/cv2")
        # C3k blocks have cv3 only when c3k=True (larger models); smaller
        # models use a plain Bottleneck (cv1+cv2 only, no cv3).
        if f"{m}.cv3.conv.weight" in self.sd:
            self.conv_bn(f"{m}.cv3", f"{o}/cv3")
        j = 0
        while f"{m}.m.{j}.cv1.conv.weight" in self.sd:
            self.conv_bn(f"{m}.m.{j}.cv1", f"{o}/inner{j}/cv1")
            self.conv_bn(f"{m}.m.{j}.cv2", f"{o}/inner{j}/cv2")
            j += 1


    def c2psa(self, pt, nn):
        self.conv_bn(f"{pt}.cv1", f"{nn}/cv1")
        self.conv_bn(f"{pt}.cv2", f"{nn}/cv2")
        self.conv_bn(f"{pt}.m.0.attn.qkv", f"{nn}/qkv")
        self.conv_bn(f"{pt}.m.0.attn.proj", f"{nn}/proj")
        self.conv_bn(f"{pt}.m.0.attn.pe", f"{nn}/pe", dw=True)
        self.conv_bn(f"{pt}.m.0.ffn.0", f"{nn}/ffn0")
        self.conv_bn(f"{pt}.m.0.ffn.1", f"{nn}/ffn1")

    def detect(self, pt):
        for i in range(3):
            self.conv_bn(f"{pt}.cv2.{i}.0", f"det{i}/cv2_0")
            self.conv_bn(f"{pt}.cv2.{i}.1", f"det{i}/cv2_1")
            self.conv_bias(f"{pt}.cv2.{i}.2", f"det{i}/cv2_2")
            self.conv_bn(f"{pt}.cv3.{i}.0.0", f"det{i}/cv3_0_dw", dw=True)
            self.conv_bn(f"{pt}.cv3.{i}.0.1", f"det{i}/cv3_0_pw")
            self.conv_bn(f"{pt}.cv3.{i}.1.0", f"det{i}/cv3_1_dw", dw=True)
            self.conv_bn(f"{pt}.cv3.{i}.1.1", f"det{i}/cv3_1_pw")
            self.conv_bias(f"{pt}.cv3.{i}.2", f"det{i}/cv3_2")

    def write(self, path):
        names = list(self.T.keys())
        header, blob, offset = {}, bytearray(), 0
        for n in names:
            arr = np.ascontiguousarray(self.T[n], dtype="<f4")
            b = arr.tobytes()
            header[n] = {"dtype": "F32", "shape": list(arr.shape),
                         "data_offsets": [offset, offset + len(b)]}
            blob += b
            offset += len(b)
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # safetensors: 8-byte little-endian header length, header, then data
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(blob)
        return len(names)


def main(weights, out):
    from ultralytics import YOLO
    model = YOLO(weights).model
    model.eval()
    model.fuse()  # fold Conv+BatchNorm into a single biased conv (inference-only)
    sd = model.state_dict()
    p = Pack(sd)

    for pt, nn in BACKBONE.items():
        p.conv_bn(pt, nn)
    for pt, nn in C3K2.items():
        p.c3k2(pt, nn)
    p.conv_bn("model.9.cv1", "m9/cv1")
    p.conv_bn("model.9.cv2", "m9/cv2")
    p.c2psa("model.10", "m10")
    p.detect("model.23")

    n = p.write(out)
    print(f"Wrote {n} tensors to nntrainer safetensors: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert YOLOv11m .pt to a single nntrainer safetensors")
    ap.add_argument("--weights", required=True,
                    help="path to the ultralytics YOLOv11m .pt checkpoint")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "res",
                                                  "yolov11m.safetensors"),
                    help="output safetensors path (default: ../res/yolov11m.safetensors)")
    args = ap.parse_args()
    main(args.weights, args.out)
