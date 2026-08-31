#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Binary-search the first op where the DSP diverges from the x86 reference.

Both runners execute ops [0, i) for the same chunk at pos=0 and dump the
output tensor of op i-1; the predicate "outputs agree" is monotone in i, so
~log2(n_ops) comparisons (9 for 451 ops) locate the first bad op.

Usage: find_divergence.py <prefix> <tokens> [--serial S] [--chunk N]
                          [--atol 0.1] [--rtol 0.1]
Prints FIRST_DIVERGENCE op=<i> kind=<k> layer=<l> max_abs=<a> max_rel=<r>
or NO_DIVERGENCE.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_RUN = os.path.join(REPO, "build_x86_hexagon", "hexagon_ref_run")
RUN_E2E = os.path.join(REPO, "tools", "hexagon", "run_e2e_test.sh")


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def list_ops(prefix):
    """[(kind, layer)] per op; the layer is derived from the op index since
    the lowering only stamps `layer` on ATTN (EMBED, then n_layers equal
    blocks, then final RMSNORM + MATMUL_LOGITS)."""
    kinds = []
    for line in sh([REF_RUN, prefix, "--list-ops"]).splitlines():
        m = re.match(r"OP (\d+) kind=(\S+)", line)
        if m:
            kinds.append(m.group(2))
    n_layers = 0
    with open(prefix + ".hexcfg") as f:
        for line in f:
            if line.startswith("n_layers="):
                n_layers = int(line.split("=")[1])
    per_layer = (len(kinds) - 3) // max(n_layers, 1)
    ops = []
    for i, k in enumerate(kinds):
        layer = -1 if i == 0 or i >= len(kinds) - 2 else (i - 1) // per_layer
        ops.append((k, layer))
    return ops


def load(path, kind):
    dt = np.float32 if kind == "MATMUL_LOGITS" else np.float16
    return np.fromfile(path, dtype=dt).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("tokens")
    ap.add_argument("--serial", default="")
    ap.add_argument("--chunk", type=int, default=0)
    # per-token int8 amplifies ~1e-4 fp16 noise 5-8x per matmul, so a tight
    # tolerance flags two correct implementations as diverged
    ap.add_argument("--atol", type=float, default=0.1)
    ap.add_argument("--rtol", type=float, default=0.1)
    args = ap.parse_args()
    chunk = ["--chunk", str(args.chunk)] if args.chunk else []
    serial = ["--serial", args.serial] if args.serial else []
    ops = list_ops(args.prefix)
    n_ops = len(ops)
    tmp = tempfile.mkdtemp(prefix="divergence_")
    n_cmp = 0

    def agree(i):
        """True when ops [0, i) produce the same output for op i-1."""
        nonlocal n_cmp
        n_cmp += 1
        ref_out = os.path.join(tmp, f"ref_{i}.bin")
        dsp_out = os.path.join(tmp, f"dsp_{i}.bin")
        out = sh([REF_RUN, args.prefix, "--tokens", args.tokens, *chunk,
                  "--dump-op", str(i), "--dump-out", ref_out])
        m = re.search(r"DUMP op=\d+ kind=(\S+) buf=(\d+) off=(\d+) bytes=(\d+)", out)
        kind, buf, off, nbytes = m.group(1), m.group(2), m.group(3), m.group(4)
        sh([RUN_E2E, args.prefix, *serial, "--", "--tokens", args.tokens,
            *chunk, "--dump-op", str(i), "--dump-buf", buf, "--dump-off",
            off, "--dump-bytes", nbytes, "--dump-out", dsp_out])
        a, b = load(ref_out, kind), load(dsp_out, kind)
        d = np.abs(a - b)
        ok = bool(np.all(d <= args.atol + args.rtol * np.abs(a)))
        stats = (float(d.max()), float((d / np.maximum(np.abs(a), 1e-6)).max()))
        print(f"cmp ops[0,{i}) op={i - 1} kind={kind} "
              f"max_abs={stats[0]:.4g} max_rel={stats[1]:.4g} -> "
              f"{'agree' if ok else 'DIFF'}", flush=True)
        return ok, stats

    # The last op writes the LOGITS buffer, which forward_debug cannot dump
    # (only WEIGHTS/KV/ACT are mapped); compare up to the final RMSNORM and
    # judge MATMUL_LOGITS by the --eval PPL of both runners.
    top = n_ops - 1 if ops[-1][0] == "MATMUL_LOGITS" else n_ops
    ok, stats = agree(top)
    if ok:
        print(f"NO_DIVERGENCE comparisons={n_cmp} (ops [0,{top}))")
        return 0
    lo, hi = 0, top  # ops[0,lo) agree, ops[0,hi) differ
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, s = agree(mid)
        if ok:
            lo = mid
        else:
            hi, stats = mid, s
    kind, layer = ops[hi - 1]
    print(f"FIRST_DIVERGENCE op={hi - 1} kind={kind} layer={layer} "
          f"max_abs={stats[0]:.4g} max_rel={stats[1]:.4g} comparisons={n_cmp}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
