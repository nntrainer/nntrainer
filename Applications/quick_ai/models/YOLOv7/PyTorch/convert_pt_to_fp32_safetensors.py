#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file convert_pt_to_fp32_safetensors.py
# @brief Convert PyTorch YOLOv7-tiny .pt checkpoint to FP32 safetensors
#        with nntrainer key naming convention.
#
#        Uses attempt_load() which fuses Conv+BN and ImplicitA/M in-place,
#        then extracts the fused weights directly.

import argparse
import json
import struct
import sys
import os

import numpy as np
import torch

# Add yolov7-test to sys.path for model imports
YOLOV7_TEST_DIR = os.environ.get('YOLOV7_TEST_DIR', '/home/seungbaek/projects/0824/l1_detector/extracted/yolov7-test')
if YOLOV7_TEST_DIR not in sys.path:
    sys.path.insert(0, YOLOV7_TEST_DIR)

from models.experimental import attempt_load


def convert(pt_path, out_path):
    """Convert PyTorch .pt to nntrainer FP32 safetensors using fused model."""
    # attempt_load calls .fuse() which fuses Conv+BN and ImplicitA/M
    model = attempt_load(pt_path, map_location='cpu')
    model.eval()
    model.float()
    
    sd = model.state_dict()
    entries = []
    
    # --- Process backbone conv layers (already fused: .conv.weight + .conv.bias) ---
    conv_keys = sorted([k for k in sd if k.endswith('.conv.weight') and k.startswith('backbone.')])
    
    for ck in conv_keys:
        prefix = ck[:-len('.conv.weight')]  # e.g., backbone.backbone.blocks.0.0
        bk = prefix + '.conv.bias'
        
        w = sd[ck].float().numpy().astype(np.float32)
        b = sd[bk].float().numpy().astype(np.float32)
        
        entries.append({
            'name': prefix + ':filter',
            'dtype': 'F32',
            'shape': list(w.shape),
            'data': w.tobytes(),
        })
        entries.append({
            'name': prefix + ':bias',
            'dtype': 'F32',
            'shape': [1] + list(b.shape) + [1, 1],
            'data': b.tobytes(),
        })
    
    # --- Process head.m.{0,1,2} (already fused with ImplicitA/M) ---
    for i in range(3):
        prefix = f'head.m.{i}'
        w = sd[f'{prefix}.weight'].float().numpy().astype(np.float32)
        b = sd[f'{prefix}.bias'].float().numpy().astype(np.float32)
        
        entries.append({
            'name': prefix + ':filter',
            'dtype': 'F32',
            'shape': list(w.shape),
            'data': w.tobytes(),
        })
        entries.append({
            'name': prefix + ':bias',
            'dtype': 'F32',
            'shape': [1] + list(b.shape) + [1, 1],
            'data': b.tobytes(),
        })
    
    # --- Write safetensors ---
    write_safetensors(out_path, entries)
    
    print(f"Wrote {len(entries)} tensors to {out_path}")
    print(f"  {len(conv_keys)} backbone conv+bn layers (fused by PyTorch)")
    print(f"  3 head conv layers (fused with ImplicitA+ImplicitM by PyTorch)")


def write_safetensors(path, entries, metadata=None):
    """Write a safetensors file with nntrainer format."""
    if metadata is None:
        metadata = {'format': 'nntrainer', 'nntr_format': 'nntr-safetensors-v1'}
    
    header = {'__metadata__': metadata}
    offset = 0
    blob = bytearray()
    
    for e in entries:
        nbytes = len(e['data'])
        header[e['name']] = {
            'dtype': e['dtype'],
            'shape': list(e['shape']),
            'data_offsets': [offset, offset + nbytes],
        }
        blob += e['data']
        offset += nbytes
    
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(blob))


def main():
    ap = argparse.ArgumentParser(
        description='Convert PyTorch YOLOv7-tiny .pt to FP32 safetensors for nntrainer')
    ap.add_argument('--pt', required=True, help='Path to .pt checkpoint')
    ap.add_argument('--out', required=True, help='Output .safetensors path')
    args = ap.parse_args()
    
    convert(args.pt, args.out)


if __name__ == '__main__':
    main()
