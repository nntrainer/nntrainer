#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Histogram of forward() RPC latency from device_test_*.log files.

Usage: python3 tools/hexagon/plot_rpc_latency.py LOG [LOG ...] [-o out.png]
Multiple logs overlay for run-to-run comparison (labels = file names).
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latencies(path):
    return [int(ln.split()[-1]) for ln in path.read_text().splitlines()
            if "forward_us" in ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=pathlib.Path)
    ap.add_argument("-o", "--out", default="logs/hexagon/rpc_latency.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4))
    for log in args.logs:
        lat = latencies(log)
        if lat:
            ax.hist(lat, bins=20, alpha=0.6, label=f"{log.name} (n={len(lat)})")
    ax.set_xlabel("forward() round-trip (us)")
    ax.set_ylabel("count")
    ax.set_title("Hexagon FastRPC dummy forward latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
