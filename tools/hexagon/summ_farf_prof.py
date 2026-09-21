#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    summ_farf_prof.py
# @brief   Summarise the per-call DSP profile lines of an HTP_PROF_FARF skel
# @author  dlwlzzero <dlwlzzero@gmail.com>
"""Summarise the per-call DSP profile lines of an HTP_PROF_FARF skel.

run_e2e_test.sh captures logcat into logs/hexagon/device_farf_<stamp>.log;
a skel built with -DHTP_PROF_FARF prints one line per forward call:

  nntr_htp: prof n=<tokens> pos=<p> ops=<n_ops> kcyc=<loop> mm8=<k> mm16=<k>
            lg=<k> attn=<k> rest=<k>

(all in kilo-pcycles). This prints, for the n=1 (decode) calls, the median
of every field in Mcycles and the share of each kind in the median step,
and for the prefill calls (n > 1) the sum. Also reports the
"stream bytes/step" line of an HTP_MM_STREAM_ONLY skel if present.

usage: summ_farf_prof.py logs/hexagon/device_farf_<stamp>.log [...]
"""
import re
import statistics
import sys

FIELDS = ("kcyc", "mm8", "mm16", "lg", "attn", "rest")
LINE = re.compile(r"nntr_htp: prof (.*)$")
KV = re.compile(r"(\w+)=(\d+)")


def summarise(path):
    decode, prefill, stream = [], [], None
    with open(path, errors="replace") as f:
        for raw in f:
            if "stream bytes/step=" in raw:
                m = re.search(r"stream bytes/step=(\d+)", raw)
                stream = int(m.group(1)) if m else stream
            m = LINE.search(raw)
            if not m:
                continue
            kv = {k: int(v) for k, v in KV.findall(m.group(1))}
            if not all(k in kv for k in FIELDS):
                continue
            (decode if kv.get("n") == 1 else prefill).append(kv)
    print(f"{path}: {len(prefill)} prefill calls, {len(decode)} decode calls")
    if stream is not None:
        print(f"  stream bytes/step = {stream:,}")
    if decode:
        med = {k: statistics.median(d[k] for d in decode) / 1000.0 for k in FIELDS}
        print("  decode median (Mcyc): " +
              " ".join(f"{k}={med[k]:.3f}" for k in FIELDS))
        loop = med["kcyc"] or 1.0
        print("  decode share of the loop: " +
              " ".join(f"{k}={100.0 * med[k] / loop:.1f}%" for k in FIELDS[1:]))
        nonmm = med["attn"] + med["rest"]
        print(f"  decode non-matmul (attn+rest) = {nonmm:.3f} Mcyc = "
              f"{100.0 * nonmm / loop:.1f}% of the loop")
    if prefill:
        tot = {k: sum(d[k] for d in prefill) / 1000.0 for k in FIELDS}
        print("  prefill sum (Mcyc): " +
              " ".join(f"{k}={tot[k]:.3f}" for k in FIELDS))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        summarise(p)
