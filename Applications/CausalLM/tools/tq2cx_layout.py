#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TQ2CX stored layouts: the v1 (row-major) <-> v2 tiled converter and a
safetensors pack re-layout tool.

A TQ2CX record holds N output rows of U = ceil(K / 32) mask units, one unit =
{uint32 pos; uint32 neg;} (8 bytes) covering 32 input taps, followed by N fp32
scales (see nntrainer/tensor/tq2cx_layout.h for the C++ side).

  "rm"     (v1) unit (n, u) at n * U + u, scales at N * U * 8; record bytes
           N * U * 8 + N * 4.
  "t2rRvV" (v2) rows in tiles of R (padded with zero units to
           Npad = ceil(N / R) * R); inside a tile, blocks of V units, block b
           holding R rows x w_b units row-major, w_b = min(V, U - b * V) (the
           tail block is packed tight):
             unit (n, u) at (n // R) * R * U + (u // V) * V * R
                            + (n % R) * w_b + u % V
           scales at Npad * U * 8, record zero-padded to a multiple of 64
           bytes.

The layout of a pack is named once per file in the safetensors header,
__metadata__["nntr_tq2cx_layout"]; a pack without the key is "rm".

Usage:
  tq2cx_layout.py selftest
  tq2cx_layout.py convert IN.safetensors OUT.safetensors [--layout t2r16v2]
  tq2cx_layout.py verify  V1.safetensors V2.safetensors [--sample 512]
"""
import argparse
import json
import random
import re
import struct
import sys

import numpy as np

META_KEY = "nntr_tq2cx_layout"


def parse_layout(name):
    """'rm' -> (0, 1); 't2rRvV' -> (R, V)"""
    if name in (None, "", "rm"):
        return 0, 1
    m = re.fullmatch(r"t2r(\d+)v(\d+)", name)
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        raise ValueError("unknown TQ2CX layout '%s'" % name)
    return int(m.group(1)), int(m.group(2))


def units(K):
    return (K + 31) // 32


def unit_index(N, U, R, V):
    """[N][U] unit positions of the layout"""
    n = np.arange(N)[:, None]
    u = np.arange(U)[None, :]
    if R == 0:
        return n * U + u
    b = u // V
    w = np.minimum(V, U - b * V)
    return (n // R) * R * U + b * V * R + (n % R) * w + (u % V)


def npad(N, R):
    return N if R == 0 else (N + R - 1) // R * R


def record_bytes(N, K, R=0, V=1):
    U = units(K)
    if R == 0:
        return N * U * 8 + N * 4
    return (npad(N, R) * U * 8 + N * 4 + 63) // 64 * 64


def v1_to_layout(rec, N, K, R=16, V=2):
    """v1 record bytes -> layout record bytes"""
    U = units(K)
    masks = np.frombuffer(rec, np.uint32, N * U * 2).reshape(N * U, 2)
    scales = bytes(rec[N * U * 8:N * U * 8 + N * 4])
    out = np.zeros((npad(N, R) * U, 2), np.uint32)
    out[unit_index(N, U, R, V).ravel()] = masks
    body = out.tobytes() + scales
    return body + b"\0" * (record_bytes(N, K, R, V) - len(body))


def layout_to_v1(rec, N, K, R=16, V=2):
    """layout record bytes -> v1 record bytes"""
    U = units(K)
    P = npad(N, R)
    lay = np.frombuffer(rec, np.uint32, P * U * 2).reshape(P * U, 2)
    masks = lay[unit_index(N, U, R, V).ravel()]
    return masks.tobytes() + bytes(rec[P * U * 8:P * U * 8 + N * 4])


def _random_v1(rng, N, K):
    U = units(K)
    a = rng.integers(0, 2**32, (N, U), dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, (N, U), dtype=np.uint64).astype(np.uint32)
    pos, neg = a & b, ~a & ~b
    if K % 32:
        keep = np.uint32((1 << (K % 32)) - 1)
        pos[:, -1] &= keep
        neg[:, -1] &= keep
    m = np.stack([pos, neg], -1)
    s = rng.random(N).astype(np.float32)
    return m.tobytes() + s.tobytes()


def selftest():
    rng = np.random.default_rng(0)
    n = 0
    for R, V in [(16, 2), (16, 4), (32, 2), (8, 1), (4, 3)]:
        for N, K in [(704, 2816), (2816, 704), (37, 416), (1, 32), (65, 224),
                     (17, 96), (33, 100)]:
            rec = _random_v1(rng, N, K)
            lay = v1_to_layout(rec, N, K, R, V)
            assert len(lay) == record_bytes(N, K, R, V)
            assert layout_to_v1(lay, N, K, R, V) == rec
            idx = unit_index(N, units(K), R, V).ravel()
            assert len(np.unique(idx)) == idx.size and idx.max() < npad(N, R) * units(K)
            n += 1
    print("selftest: %d layout x shape round trips OK" % n)


def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def tq2_dims(entry):
    """(N, K) of a TQ2CX entry ([1, E, K, N] nntr_shape, E == 1)"""
    shp = entry["nntr_shape"]
    if len(shp) != 4 or shp[0] != 1 or shp[1] != 1:
        raise ValueError("TQ2CX entry with nntr_shape %s: one expert per "
                         "entry expected" % shp)
    return shp[3], shp[2]


def header_bytes(hdr, align=64):
    """serialize, padded with spaces so the data section starts aligned"""
    s = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-(8 + len(s))) % align
    return s + b" " * pad


def convert(src, dst, layout):
    R, V = parse_layout(layout)
    if R == 0:
        raise ValueError("convert writes a tiled layout")
    hdr, base = read_header(src)
    meta = dict(hdr.get("__metadata__", {}))
    old = meta.get(META_KEY, "rm")
    if old != "rm":
        raise ValueError("%s is already in layout %s" % (src, old))
    items = sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                   key=lambda kv: kv[1]["data_offsets"][0])
    for k, v in items:
        if v.get("nntr_dtype") == "TQ2CX":
            N, K = tq2_dims(v)
            nb = v["data_offsets"][1] - v["data_offsets"][0]
            if nb != record_bytes(N, K) or record_bytes(N, K, R, V) != nb:
                raise ValueError(
                    "%s: %d bytes, v1 %d, %s %d -- a size change would move "
                    "every offset; not supported by this tool" %
                    (k, nb, record_bytes(N, K), layout,
                     record_bytes(N, K, R, V)))
    meta[META_KEY] = layout
    out_hdr = {"__metadata__": meta}
    for k, v in hdr.items():
        if k != "__metadata__":
            out_hdr[k] = v
    hb = header_bytes(out_hdr)
    n_tq2 = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        pos = 0
        for k, v in items:
            s, e = v["data_offsets"]
            if s != pos:
                raise ValueError("gap/overlap before %s (%d vs %d)" % (k, s, pos))
            fi.seek(base + s)
            data = fi.read(e - s)
            if len(data) != e - s:
                raise IOError("short read at %s" % k)
            if v.get("nntr_dtype") == "TQ2CX":
                N, K = tq2_dims(v)
                data = v1_to_layout(data, N, K, R, V)
                n_tq2 += 1
            fo.write(data)
            pos = e
    print("%s: %d TQ2CX records -> %s, %d tensors, data %d bytes" %
          (dst, n_tq2, layout, len(items), pos))


def verify(v1, v2, sample):
    h1, b1 = read_header(v1)
    h2, b2 = read_header(v2)
    layout = h2.get("__metadata__", {}).get(META_KEY, "rm")
    R, V = parse_layout(layout)
    k1 = [k for k in h1 if k != "__metadata__"]
    k2 = [k for k in h2 if k != "__metadata__"]
    assert sorted(k1) == sorted(k2), "key sets differ"
    m1 = dict(h1.get("__metadata__", {}))
    m2 = dict(h2.get("__metadata__", {}))
    m2.pop(META_KEY, None)
    assert m1 == m2, "metadata differs beyond the layout key"
    # full key / offset audit of the new file: contiguous, no gaps / overlaps
    spans = sorted((h2[k]["data_offsets"][0], h2[k]["data_offsets"][1], k)
                   for k in k2)
    pos = 0
    for s, e, k in spans:
        assert s == pos, "gap/overlap at %s (%d vs %d)" % (k, s, pos)
        assert e >= s
        assert h1[k] == h2[k], "entry %s changed" % k
        pos = e
    import os
    assert os.path.getsize(v2) == b2 + pos, "file size vs data section"
    print("audit: %d keys, contiguous data section of %d bytes, header %d"
          % (len(k2), pos, b2))
    tq = [k for k in k2 if h2[k].get("nntr_dtype") == "TQ2CX"]
    other = [k for k in k2 if h2[k].get("nntr_dtype") != "TQ2CX"]
    rnd = random.Random(1003)
    pick = rnd.sample(tq, min(sample, len(tq)))
    with open(v1, "rb") as f1, open(v2, "rb") as f2:
        for k in pick:
            s, e = h1[k]["data_offsets"]
            f1.seek(b1 + s)
            a = f1.read(e - s)
            f2.seek(b2 + s)
            b = f2.read(e - s)
            N, K = tq2_dims(h1[k])
            assert layout_to_v1(b, N, K, R, V) == a, "round trip " + k
            assert v1_to_layout(a, N, K, R, V) == b, "forward " + k
        # every non-TQ2CX tensor verbatim
        for k in other:
            s, e = h1[k]["data_offsets"]
            f1.seek(b1 + s)
            f2.seek(b2 + s)
            assert f1.read(e - s) == f2.read(e - s), "changed " + k
    print("round trip: %d / %d TQ2CX records OK (%s); %d other tensors "
          "verbatim" % (len(pick), len(tq), layout, len(other)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("convert")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--layout", default="t2r16v2")
    v = sub.add_parser("verify")
    v.add_argument("v1")
    v.add_argument("v2")
    v.add_argument("--sample", type=int, default=512)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "convert":
        convert(a.src, a.dst, a.layout)
    else:
        verify(a.v1, a.v2, a.sample)


if __name__ == "__main__":
    sys.exit(main())
