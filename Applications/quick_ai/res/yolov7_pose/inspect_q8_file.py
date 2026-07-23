#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Inspect an nntrainer-produced (quantized) .safetensors file for corruption.

nntr_quantize writes:
  * plain tensors (head weights, biases, non-eligible convs) as "F32"/"F16"
  * Q8_0 conv weights as a "U8" byte blob with an "nntr_dtype":"Q8_0" extension,
    physically laid out as block_q8_0x4 super-blocks (4 fp16 scales + 128 int8,
    the 4-column interleave produced by repack_q8_0 at save time).

This checks every tensor for non-finite values *without* nntrainer or a rebuild:
  - F32/F16 tensors: report nan/inf/min/max over the raw values.
  - Q8_0 tensors: decode the fp16 block scales + int8 payload and report the
    dequantized min/max plus any non-finite scale, so a bad quantize/repack or a
    truncated blob is caught directly on the file.

Usage:
  python3 inspect_q8_file.py yolov7_pose_q8_0.safetensors
  python3 inspect_q8_file.py yolov7_pose_q8_0.safetensors --only head   # substr
"""
import argparse
import json
import struct
import sys

import numpy as np


def load_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        base = 8 + hlen
    return header, base


def read_span(path, base, off):
    with open(path, "rb") as f:
        f.seek(base + off[0])
        return f.read(off[1] - off[0])


def scan_q8_0x4(raw):
    """Validate a block_q8_0x4 blob straight from its bytes (shape-independent).

    Each 136-byte super-block = 4 fp16 scales + 128 int8. Reports the number of
    non-finite scales and the dequantized value range across the whole blob.
    """
    sb_bytes = 8 + 128
    if len(raw) % sb_bytes != 0:
        return None, f"not a multiple of super-block (136B): {len(raw)} bytes"
    nsb = len(raw) // sb_bytes
    d = np.frombuffer(raw, dtype=np.uint8).reshape(nsb, sb_bytes)
    scales = (
        np.frombuffer(d[:, :8].tobytes(), dtype=np.float16)
        .astype(np.float32)
        .reshape(nsb, 4)
    )
    qs = d[:, 8:].astype(np.int8).astype(np.float32)  # (nsb, 128)
    bad = int((~np.isfinite(scales)).sum())
    # dequant magnitude bound: |val| <= |q| * |scale|; use per-superblock/per-row
    finite = np.isfinite(scales)
    smax = float(np.abs(scales[finite]).max()) if finite.any() else float("nan")
    qmax = float(np.abs(qs).max())
    approx_max = smax * qmax
    note = f"scales:bad={bad} |scale|max={smax:.4g} |q|max={qmax:.0f} |val|<={approx_max:.4g}"
    ok = bad == 0
    return ok, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--only", default="", help="only tensors whose name contains this")
    args = ap.parse_args()

    header, base = load_header(args.file)
    names = [k for k in header if k != "__metadata__"]
    names.sort()

    n_bad = 0
    for name in names:
        if args.only and args.only not in name:
            continue
        e = header[name]
        dt = e.get("nntr_dtype", e["dtype"])
        shape = e.get("shape", [])
        off = e["data_offsets"]
        raw = read_span(args.file, base, off)
        note = ""
        if dt in ("F32", "FP32"):
            v = np.frombuffer(raw, dtype=np.float32)
        elif dt in ("F16", "FP16"):
            v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "Q8_0":
            ok, note = scan_q8_0x4(raw)
            flag = "ok" if ok else "!!"
            if not ok:
                n_bad += 1
            print(f"  {flag} {name:50s} {dt:6s} {str(shape):20s} bytes={len(raw)} {note}")
            continue
        else:
            print(f"  -- {name:50s} {dt:6s} {str(shape):20s} (skipped)")
            continue

        nnan = int(np.isnan(v).sum())
        ninf = int(np.isinf(v).sum())
        fin = v[np.isfinite(v)]
        mn = float(fin.min()) if fin.size else float("nan")
        mx = float(fin.max()) if fin.size else float("nan")
        flag = "!!" if (nnan or ninf or note) else "ok"
        if nnan or ninf or note:
            n_bad += 1
        print(
            f"  {flag} {name:50s} {dt:6s} {str(shape):20s} "
            f"nan={nnan} inf={ninf} min={mn:+.4g} max={mx:+.4g} {note}"
        )

    print(f"\n{'CORRUPT' if n_bad else 'CLEAN'}: {n_bad} tensor(s) with non-finite/bad data")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
