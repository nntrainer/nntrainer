#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verdict for a device_test_*.log from run_device_test.sh.

Usage: python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_X.log
Exit code 0 iff the run passed. Prints a marker checklist and forward()
latency stats (one RPC per forward by construction).
"""
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

    lat = [int(ln.split()[-1]) for ln in lines if "forward_us" in ln]
    if lat:
        print(f"forward RPC x{len(lat)}: min {min(lat)} us, "
              f"median {statistics.median(lat)} us, max {max(lat)} us")
    else:
        print("[MISSING] forward_us samples")
        ok = False

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
