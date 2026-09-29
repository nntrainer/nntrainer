#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    check_rpc_log.py
# @brief   Verdict for a device_test_*.log from run_device_test.sh
# @author  dlwlzzero <dlwlzzero@gmail.com>
"""Verdict for a device_test_*.log from run_device_test.sh.

Usage: python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_X.log
Exit code 0 iff the run passed. Prints a marker checklist and forward()
latency stats (one RPC per forward by construction), plus the per-memory
transport rows (`forward_full_us mem=<m> <us>`, full-vocab logits in
malloc / rpcmem / static memory) as an informational table: the delta
against the 8-float median is the cost of returning the logits.
"""
import re
import statistics
import sys

REQUIRED = [
    "RPC_TEST open ok",
    "RPC_TEST rpcmem ok",
    "RPC_TEST bad-version rejected ok",
    "RPC_TEST init ok",
    "RPC_TEST pattern ok",
    "RPC_TEST PASS",
]


def main(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    ok = True
    for marker in REQUIRED:
        found = any(marker in ln for ln in lines)
        print(f"[{'ok' if found else 'MISSING'}] {marker}")
        ok &= found

    fails = [ln for ln in lines if "RPC_TEST FAIL" in ln]
    for ln in fails:
        print(ln)
    ok &= not fails

    lat = [int(ln.split()[-1]) for ln in lines if "RPC_TEST forward_us " in ln]
    if lat:
        print(f"forward RPC x{len(lat)}: min {min(lat)} us, "
              f"median {statistics.median(lat)} us, max {max(lat)} us")
    else:
        print("[MISSING] forward_us samples")
        ok = False

    # Informational: full-vocab logits per host memory kind (#24). Missing
    # rows (e.g. an SDK without FASTRPC_MAP_STATIC skips mem=static) are
    # not a failure; the marker list above is the pass/fail contract. The
    # "-forward_us(8)" delta is an upper bound on the return cost: it also
    # contains the DSP's scalar fill of 151,936 elements (a software modulo
    # each, executor.c), which the dummy path does not report as pcycles;
    # the fill is identical across rows, so row-to-row differences are the
    # clean transport number.
    full = {}
    for ln in lines:
        m = re.search(r"RPC_TEST forward_full_us mem=(\w+) (\d+)$", ln)
        if m:
            full.setdefault(m.group(1), []).append(int(m.group(2)))
    for mem, v in full.items():
        med = statistics.median(v)
        delta = f", -forward_us(8) {med - statistics.median(lat):+.1f} us" if lat else ""
        print(f"forward_full mem={mem} x{len(v)}: min {min(v)} us, "
              f"median {med} us, max {max(v)} us{delta}")

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
