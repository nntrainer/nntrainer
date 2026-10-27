#!/usr/bin/env python3
"""
@file run_pytorch.py
@brief Run PyTorch YOLOv11m and print final detections, AND write the exact 832x832
input tensor as a .bin so the nntrainer example can run on identical bytes
(apples-to-apples comparison).

Examples:
  # real image:
  python run_pytorch.py --weights v11m_832rect_best.pt --image cat.jpeg
  # deterministic seed-42 noise (matches res/input_832.bin):
  python run_pytorch.py --weights v11m_832rect_best.pt

It writes the preprocessed input to --save-input (default ../res/input_run.bin)
and prints detections [x1,y1,x2,y2,conf,cls] (xyxy pixels at the 832 scale).
Run the nntrainer side on the SAME input with:  ./run_nntrainer.sh ../res/input_run.bin
"""
import argparse
import os

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REG_MAX, NC, STRIDES = 16, 1, (8, 16, 32)
GRID = {8: 104, 16: 52, 32: 26}  # for 832 input


def letterbox(path, size=832, pad=114):
    import cv2
    img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    t = torch.from_numpy(out).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.contiguous()


def make_anchors():
    pts, strd = [], []
    for s in STRIDES:
        g = GRID[s]
        sx = (torch.arange(g) + 0.5)
        sy = (torch.arange(g) + 0.5)
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        pts.append(torch.stack((xx.reshape(-1), yy.reshape(-1)), 1))
        strd.append(torch.full((g * g, 1), float(s)))
    return torch.cat(pts), torch.cat(strd)  # [N,2], [N,1]


def decode(raws):
    # raws[i]: [1, 64+NC, H, W]
    box = torch.cat([r[:, :4 * REG_MAX].reshape(1, 4 * REG_MAX, -1) for r in raws], 2)
    cls = torch.cat([r[:, 4 * REG_MAX:].reshape(1, NC, -1) for r in raws], 2)
    b, _, n = box.shape
    proj = torch.arange(REG_MAX, dtype=torch.float32)
    dist = box.view(b, 4, REG_MAX, n).softmax(2).mul(proj.view(1, 1, -1, 1)).sum(2)  # [1,4,N]
    anchors, strd = make_anchors()
    anchors = anchors.t().unsqueeze(0)  # [1,2,N]
    lt, rb = dist[:, :2], dist[:, 2:]
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    cxcy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    xywh = torch.cat([cxcy, wh], 1) * strd.t().unsqueeze(0)  # [1,4,N] px@832
    score = cls.sigmoid()
    return xywh, score  # [1,4,N], [1,NC,N]


def nms(xywh, score, conf=0.25, iou=0.70, max_det=300):
    import torchvision
    cx, cy, w, h = xywh[0]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], 1)  # [N,4]
    conf_t, cls_t = score[0].max(0)  # [N], [N]
    keep0 = conf_t > conf
    boxes, conf_t, cls_t = boxes[keep0], conf_t[keep0], cls_t[keep0]
    if boxes.numel() == 0:
        return []
    offset = cls_t.float().unsqueeze(1) * 7680.0
    keep = torchvision.ops.nms(boxes + offset, conf_t, iou)[:max_det]
    return [(*boxes[k].tolist(), float(conf_t[k]), int(cls_t[k])) for k in keep]


def main(weights, image, save_input, conf, iou):
    from ultralytics import YOLO
    m = YOLO(weights).model
    m.eval()

    if image:
        x = letterbox(image)
        print(f"input: letterboxed {image} -> [1,3,832,832]")
    else:
        torch.manual_seed(42)
        x = torch.randn(1, 3, 832, 832)
        print("input: fixed seed-42 noise (matches res/input_832.bin)")

    x.detach().cpu().numpy().astype("<f4").tofile(save_input)
    print(f"saved input bin for nntrainer: {save_input}")

    raws = {}

    def hook(mod, inp, out):
        feats = inp[0]
        for i, xi in enumerate(feats):
            raws[i] = torch.cat([mod.cv2[i](xi), mod.cv3[i](xi)], 1).detach()

    h = m.model[-1].register_forward_hook(hook)
    with torch.no_grad():
        m(x)
    h.remove()

    xywh, score = decode([raws[0], raws[1], raws[2]])
    dets = nms(xywh, score, conf, iou)
    print(f"\nPyTorch detections (conf>={conf}, iou={iou}): {len(dets)}")
    for i, d in enumerate(dets):
        print("  [%d] (%.3f, %.3f, %.3f, %.3f) conf=%.6f cls=%d" % (i, *d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyTorch YOLOv11m run + input dump")
    ap.add_argument("--weights", required=True, help="ultralytics .pt checkpoint")
    ap.add_argument("--image", default=None, help="image path (omit = seed-42 noise)")
    ap.add_argument("--save-input", default=os.path.join(_HERE, "..", "res", "input_run.bin"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.70)
    a = ap.parse_args()
    main(a.weights, a.image, a.save_input, a.conf, a.iou)
