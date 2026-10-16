#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recover a pre-rotated ternary Gemma4 checkpoint's rotation signs from its
expert weights, against an unrotated pack of the same model (PR 4428).

The runtime rotates an expert input as x <- x (H44^T x H_p) / sqrt(n), so the
converter must have stored W' = (H44 x H_p) W / sqrt(n) per expert weight
(W is [K, N], K the input width). A seeded rotation (rotation_meta.json,
seed 42) adds one sign per input dimension: W' = diag(s) (H44 x H_p) W.
Row k of (H44 x H_p) W_unrotated and row k of W' then correlate with sign
s[k], |r| ~ 0.75 through the ternary quantization of both. This script
reads one expert of each layer from both packs (gate_up: K = hidden, down:
K = moe_intermediate), prints the correlation and the sign agreement between
two experts, and writes the nntr_hadamard_signs_v1 sidecar the runtime
takes through nntr_config "moe_hadamard_signs".

Record offsets come from the runtime: run each pack once with
NNTR_MOE_DESC_TRACE=1 (any short prompt) and save the [MoE-DESC] lines.

  recover_hadamard_signs.py --rotated BIN --rotated-desc TXT \\
      --plain BIN --plain-desc TXT --out rotation_signs.json [--layers 0 29]
  (paths may be adb:<device>:<path>, read with dd through adb exec-out)
"""
import argparse
import json
import re
import subprocess
import sys

import numpy as np


def wh_unpack(b, K, N):
    """QS4CX_WH int4 codes (htp_wh_layout.h whUnpack) -> [K, N] int."""
    T = 32
    kt, nt = K // T, N // T
    tiles = np.frombuffer(b[:kt * nt * 512], np.uint8).reshape(kt, nt, 512)
    r = np.arange(T)[:, None]
    c = np.arange(T)[None, :]
    sl = (r // 8) * 256 + c * 8 + (r % 4) * 2 + ((r // 4) % 2)
    nib = (tiles[:, :, sl // 2] >> (4 * (sl % 2))) & 0xF
    q = nib.astype(np.int16)
    q[q > 7] -= 16
    return q.transpose(0, 2, 1, 3).reshape(K, N)


def tl2_unpack(b, K, N):
    """Native TL2 trits (htp_tl2_native_packing.h) -> [K, N] in {-1, 0, 1}."""
    blocks, P = (K + 23) // 24, (N + 127) // 128
    d = np.frombuffer(b[:P * blocks * 640], np.uint8).reshape(P, blocks, 640)
    q = np.zeros((K, N), np.int16)
    j = np.arange(128)
    n_of_j = (j % 4) * 32 + j // 4
    for p in range(P):
        cols = p * 128 + n_of_j
        ok = cols < N
        for kb in range(blocks):
            blk = d[p, kb]
            for g in range(8):
                mag = (blk[(g // 2) * 128 + j] >> (4 * (g % 2))) & 0xF
                sgn = (blk[512 + j] >> g) & 1
                v = np.where(sgn == 1, -mag.astype(np.int16), mag.astype(np.int16))
                trits = []
                for _ in range(3):  # balanced ternary, least significant first
                    r = ((v % 3) + 3) % 3
                    r = np.where(r == 2, -1, r)
                    trits.append(r)
                    v = (v - r) // 3
                for t, w in enumerate(reversed(trits)):
                    k = kb * 24 + 3 * g + t
                    if k < K:
                        q[k, cols[ok]] = w[ok]
    return q


def paley(i, j):
    if j == 0 or i == j:
        return 1
    if i == 0:
        return -1
    return 1 if pow((i - j) % 43, 21, 43) == 1 else -1


def sylvester(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


H44 = np.array([[paley(i, j) for j in range(44)] for i in range(44)], np.float32)


def read_range(path, off, length):
    if path.startswith("adb:"):
        _, serial, f = path.split(":", 2)
        blk, cnt = off // 4096, (off % 4096 + length + 4095) // 4096
        raw = subprocess.run(
            ["adb", "-s", serial, "exec-out",
             f"dd if={f} bs=4096 skip={blk} count={cnt} 2>/dev/null"],
            capture_output=True, check=True).stdout
        return raw[off % 4096:off % 4096 + length]
    with open(path, "rb") as fh:
        fh.seek(off)
        return fh.read(length)


def descs(path):
    out = {}
    pat = re.compile(r"layer=(\d+) expert=(\d+) K=(\d+) inter=(\d+) N_out=(\d+) "
                     r"w_bits=(\d+) affine=(\d) off_gu=(\d+) off_dn=(\d+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            v = [int(x) for x in m.groups()]
            out[(v[0], v[1])] = dict(K=v[2], inter=v[3], N_out=v[4], bits=v[5],
                                     affine=v[6], off_gu=v[7], off_dn=v[8])
    return out


def dequant(path, off, bits, affine, k, n):
    if bits == 258:  # native TL2 (HTP_TL2_WEIGHT_BITS)
        nb = ((k + 23) // 24) * ((n + 127) // 128) * 640
        b = read_range(path, off, nb + 8 * n)
        return tl2_unpack(b, k, n).astype(np.float32) * np.frombuffer(b[nb:nb + 4 * n], np.float32)
    codes = k * n // 2
    b = read_range(path, off, codes + (12 if affine else 8) * n)
    q = wh_unpack(b, k, n)
    tail = b[codes:]
    scale = np.frombuffer(tail[:4 * n], np.float32)
    offset = np.frombuffer(tail[8 * n:12 * n], np.int32) if affine else 0
    return (q + offset).astype(np.float32) * scale


def row_corr(C, W):
    Cc = C - C.mean(1, keepdims=True)
    Wc = W - W.mean(1, keepdims=True)
    return (Cc * Wc).sum(1) / np.sqrt((Cc ** 2).sum(1) * (Wc ** 2).sum(1) + 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotated", required=True)
    ap.add_argument("--rotated-desc", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--plain-desc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs=2, type=int, default=[0, 29])
    a = ap.parse_args()
    dr, dp = descs(a.rotated_desc), descs(a.plain_desc)
    signs_in = signs_dn = None
    weak = 0
    for layer in range(a.layers[0], a.layers[1] + 1):
        per = []
        for e in (0, 1):
            r, p = dr[(layer, e)], dp[(layer, e)]
            K, I, N = r["K"], r["inter"], r["N_out"]
            M_in = (np.kron(H44, sylvester(K // 44)) / np.sqrt(K)).astype(np.float32)
            M_dn = (np.kron(H44, sylvester(I // 44)) / np.sqrt(I)).astype(np.float32)
            Wr_gu = dequant(a.rotated, r["off_gu"], r["bits"], r["affine"], K, 2 * I)
            Wp_gu = dequant(a.plain, p["off_gu"], p["bits"], p["affine"], K, 2 * I)
            Wr_dn = dequant(a.rotated, r["off_dn"], r["bits"], r["affine"], I, N)
            Wp_dn = dequant(a.plain, p["off_dn"], p["bits"], p["affine"], I, N)
            per.append((row_corr(M_in @ Wp_gu, Wr_gu), row_corr(M_dn @ Wp_dn, Wr_dn)))
        s_in, s_dn = np.sign(per[0][0]), np.sign(per[0][1])
        agree = int((s_in == np.sign(per[1][0])).sum()), int((s_dn == np.sign(per[1][1])).sum())
        weak += int((np.abs(per[0][0]) < 0.2).sum() + (np.abs(per[0][1]) < 0.2).sum())
        print(f"layer {layer:2d}: |r_in| {np.abs(per[0][0]).mean():.3f} |r_down| "
              f"{np.abs(per[0][1]).mean():.3f} experts agree {agree[0]}/{len(s_in)} "
              f"{agree[1]}/{len(s_dn)}", flush=True)
        if signs_in is None:
            signs_in, signs_dn = s_in, s_dn
        elif not (np.array_equal(signs_in, s_in) and np.array_equal(signs_dn, s_dn)):
            sys.exit(f"layer {layer}: its signs differ from layer {a.layers[0]}'s; "
                     "the runtime takes one vector pair for every layer")
    if weak:
        sys.exit(f"{weak} rows correlate below 0.2: the orientation or the pack is off")
    json.dump({"format": "nntr_hadamard_signs_v1",
               "hadamard": "x <- (x (H44^T x H_p) / sqrt(n)) * s, H44 Paley as in SpinQuant",
               "in": signs_in.astype(int).tolist(),
               "down": signs_dn.astype(int).tolist()}, open(a.out, "w"))
    print("wrote", a.out, "in -1:", int((signs_in < 0).sum()), "down -1:", int((signs_dn < 0).sum()))


if __name__ == "__main__":
    main()
