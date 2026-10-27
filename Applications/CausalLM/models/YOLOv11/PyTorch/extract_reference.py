#!/usr/bin/env python3
"""
@file extract_reference.py
@brief Extract PyTorch YOLOv11m reference outputs for nntrainer verification.
Fixed seed input → raw Detect head logits (before DFL/sigmoid).
"""
import torch
import numpy as np
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
# Resolved from CLI args in __main__ (see bottom of file).
WEIGHTS = None
OUT_DIR = None

def save_bin(tensor, path):
    arr = tensor.detach().cpu().float().numpy()
    arr.tofile(path)
    print(f"  saved {path} shape={arr.shape} dtype={arr.dtype}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    m = model.model
    m.eval()

    # Fixed input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 832, 832)
    save_bin(x, os.path.join(OUT_DIR, 'input_832.bin'))
    print(f"Input shape: {x.shape}")

    # Hook to capture Detect head raw inputs (before DFL/sigmoid)
    # model.23 is the Detect layer; its forward receives [P3, P4, P5]
    # We want the raw cv2/cv3 outputs per scale (before decode)
    detect_raw_inputs = {}
    detect_raw_outputs = {}

    def hook_detect_input(module, inputs, output):
        # inputs is tuple of feature maps [P3, P4, P5]
        for i, inp in enumerate(inputs[0] if isinstance(inputs[0], (list, tuple)) else inputs):
            detect_raw_inputs[i] = inp.detach().clone()  # clone() prevents aliasing from in-place ops
        # Capture cv2/cv3 raw outputs before decode
        # We manually run cv2/cv3 here
        detect = module
        for i, xi in enumerate(detect_raw_inputs.values()):
            box = detect.cv2[i](xi)  # [B, 4*reg_max, H, W]
            cls = detect.cv3[i](xi)  # [B, nc, H, W]
            raw = torch.cat([box, cls], dim=1)
            detect_raw_outputs[i] = raw.detach().clone()  # clone() prevents aliasing from in-place ops
            print(f"  P{i+3} input shape: {xi.shape}, box: {box.shape}, cls: {cls.shape}")

    # Find model.23 (Detect)
    detect_layer = m.model[-1]
    hook = detect_layer.register_forward_hook(hook_detect_input)

    # Also capture intermediate feature maps for block-level verification
    feature_maps = {}
    hooks = []

    # Key checkpoints: after each major block
    key_layers = {
        0: 'conv0', 1: 'conv1', 2: 'c3k2_2', 3: 'conv3',
        4: 'c3k2_4', 5: 'conv5', 6: 'c3k2_6', 7: 'conv7',
        8: 'c3k2_8', 9: 'sppf_9', 10: 'c2psa_10',
        16: 'head_p3_16', 19: 'head_p4_19', 22: 'head_p5_22',
    }

    def make_hook(name):
        def h(module, inp, output):
            if isinstance(output, torch.Tensor):
                feature_maps[name] = output.detach().clone()  # clone() prevents aliasing from in-place ops
        return h

    for idx, name in key_layers.items():
        layer = m.model[idx]
        hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = m(x)

    hook.remove()
    for h in hooks:
        h.remove()

    # Save raw detect outputs
    for i, raw in detect_raw_outputs.items():
        scale_name = ['p3', 'p4', 'p5'][i]
        save_bin(raw, os.path.join(OUT_DIR, f'ref_{scale_name}.bin'))

    # Save key feature maps
    for name, feat in feature_maps.items():
        save_bin(feat, os.path.join(OUT_DIR, f'ref_{name}.bin'))

    # Print summary
    print("\n=== Reference output summary ===")
    print(f"Input: {x.shape}")
    for i, raw in detect_raw_outputs.items():
        scale_name = ['P3', 'P4', 'P5'][i]
        print(f"Detect {scale_name} raw: {raw.shape}  (box+cls concat)")
    for name, feat in feature_maps.items():
        print(f"  {name}: {feat.shape}")

    print(f"\nAll saved to: {OUT_DIR}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract YOLOv11m PyTorch reference outputs for verification')
    ap.add_argument('--weights', required=True,
                    help='path to the ultralytics YOLOv11m .pt checkpoint')
    ap.add_argument('--out', default=os.path.join(_HERE, '..', 'res'),
                    help='output dir for reference .bin files (default: ../res)')
    args = ap.parse_args()
    WEIGHTS = args.weights
    OUT_DIR = args.out
    main()
