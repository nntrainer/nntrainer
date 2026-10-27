#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   compare_safetensors_q40.py
@brief  Byte-equivalence check between two nntrainer safetensors files.

Verifies that the C++ `nntr_quantize` output matches the reference python
quantizer output (quantize_q4_0_conv.py) tensor-for-tensor. For every tensor
present in both files it compares the raw data blob byte-for-byte and reports
the per-tensor verdict, with special attention to Q4_0 (nntr_dtype) tensors.

Usage:
  python compare_safetensors_q40.py A.safetensors B.safetensors
"""
import json
import struct
import sys


def parse(path):
    with open(path, "rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hsize).decode("utf-8"))
        data = f.read()
    return header, data


def main():
    if len(sys.argv) != 3:
        print("usage: compare_safetensors_q40.py A.safetensors B.safetensors")
        sys.exit(2)
    ha, da = parse(sys.argv[1])
    hb, db = parse(sys.argv[2])

    names_a = {k for k in ha if k != "__metadata__"}
    names_b = {k for k in hb if k != "__metadata__"}

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    if only_a:
        print(f"[only in A] {len(only_a)}: {only_a[:8]}{'...' if len(only_a) > 8 else ''}")
    if only_b:
        print(f"[only in B] {len(only_b)}: {only_b[:8]}{'...' if len(only_b) > 8 else ''}")

    n_q40 = 0
    n_q40_match = 0
    n_other = 0
    n_other_match = 0
    mismatches = []

    for name in common:
        ea, eb = ha[name], hb[name]
        oa, ob = ea["data_offsets"], eb["data_offsets"]
        ba = da[oa[0]:oa[1]]
        bb = db[ob[0]:ob[1]]
        is_q40 = ea.get("nntr_dtype") == "Q4_0" or eb.get("nntr_dtype") == "Q4_0"
        match = (ba == bb)
        if is_q40:
            n_q40 += 1
            n_q40_match += int(match)
        else:
            n_other += 1
            n_other_match += int(match)
        if not match:
            # find first differing byte
            first = next((i for i in range(min(len(ba), len(bb))) if ba[i] != bb[i]), None)
            mismatches.append(
                (name, "Q4_0" if is_q40 else ea.get("nntr_dtype", ea.get("dtype")),
                 len(ba), len(bb), first,
                 ea.get("nntr_shape", ea.get("shape")),
                 eb.get("nntr_shape", eb.get("shape"))))

    print(f"\nQ4_0 tensors:  {n_q40_match}/{n_q40} byte-identical")
    print(f"other tensors: {n_other_match}/{n_other} byte-identical")
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for name, dt, la, lb, first, sa, sb in mismatches[:40]:
            print(f"  {name} [{dt}] lenA={la} lenB={lb} firstDiff={first} shapeA={sa} shapeB={sb}")
    ok = (not only_a and not only_b and not mismatches)
    print("\nRESULT:", "IDENTICAL ✓" if ok else "DIFFERENCES FOUND ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
