#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete, interleaved Hadamard benchmark runs (stdlib only)."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

NAMES = ["off", "up", "down", "both", "both_scalar", "up_hmx", "down_hmx", "both_hmx"]


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(path):
    lines = Path(path).read_text().splitlines()
    checks = [line for line in lines if line.startswith("CHECK ")]
    if len(checks) != 1:
        raise ValueError(f"{path}: missing scalar/HVX parity check")
    start = next((i for i, line in enumerate(lines) if line.startswith("M,bits,rep,")), None)
    if start is None:
        raise ValueError(f"{path}: no timing rows")
    rows = list(csv.DictReader(lines[start:]))
    ids = {int(row["variant"]) for row in rows}
    if ids not in (set(range(5)), set(range(8))):
        raise ValueError(f"{path}: missing/unknown variants")
    hmx_checks = [line for line in lines if line.startswith("CHECK_HMX")]
    if len(ids) == 8 and not any(line.startswith("CHECK_HMX ") for line in hmx_checks):
        raise ValueError(f"{path}: missing HMX parity check")
    samples = {v: {} for v in sorted(ids)}
    shapes = set()
    for row in rows:
        v, rep = int(row["variant"]), int(row["rep"])
        shapes.add((int(row["M"]), int(row["bits"])))
        if v not in samples or rep in samples[v]:
            raise ValueError(f"{path}: duplicate/unknown variant")
        value = float(row["dsp_us"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: invalid DSP time")
        samples[v][rep] = value
    if len(shapes) != 1 or not samples[0]:
        raise ValueError(f"{path}: inconsistent/empty run")
    reps = set(samples[0])
    if any(set(v) != reps for v in samples.values()) or reps != set(range(len(reps))):
        raise ValueError(f"{path}: incomplete repetition")
    baseline = statistics.median(samples[0].values())
    if min(samples[0].values()) <= 0:
        raise ValueError(f"{path}: invalid baseline timing")
    variants = {}
    for v, name in enumerate(NAMES[:len(samples)]):
        values = list(samples[v].values())
        deltas = [samples[v][r] - samples[0][r] for r in reps]
        med = statistics.median(values)
        variants[name] = {
            "n": len(values),
            "dsp_median_us": med,
            "dsp_min_us": min(values),
            "dsp_max_us": max(values),
            "dsp_p25_us": percentile(values, 0.25),
            "dsp_p75_us": percentile(values, 0.75),
            "delta_medians_us": med - baseline,
            "delta_medians_pct": 100 * (med / baseline - 1),
            "paired_median_delta_us": statistics.median(deltas),
            "paired_delta_p25_us": percentile(deltas, 0.25),
            "paired_delta_p75_us": percentile(deltas, 0.75),
            "paired_median_delta_pct": statistics.median(
                100 * (samples[v][r] / samples[0][r] - 1) for r in reps
            ),
        }
    tokens, bits = shapes.pop()
    return {"file": str(path), "tokens": tokens, "weight_bits": bits,
            "check": checks[0], "hmx_checks": hmx_checks,
            "config": [line for line in lines if line.startswith("CONFIG ")],
            "variants": variants}


def pool_ffn(paths):
    """Sum the medians of the two FFN kernels; this is not a full-model timer."""
    samples = {2: [], 4: []}
    for path in paths:
        summary = summarize(path)
        bits = summary["weight_bits"]
        if (summary["tokens"] != 1024 or bits not in samples
                or summary["config"] != ["CONFIG pre_norm=1"]
                or "both_hmx" not in summary["variants"]):
            raise ValueError(f"{path}: expected 1024-token RMSNorm HMX run")
        lines = Path(path).read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("M,bits,rep,"))
        rows = {}
        for row in csv.DictReader(lines[start:]):
            rows.setdefault(int(row["rep"]), {})[int(row["variant"])] = float(row["dsp_us"])
        samples[bits].extend((v[0], v[7]) for v in rows.values())
    if not all(samples.values()):
        raise ValueError("pool requires both expert-2bit and dense-4bit runs")

    def estimate(data):
        baseline = sum(statistics.median(x[0] for x in v) for v in data.values())
        hmx = sum(statistics.median(x[1] for x in v) for v in data.values())
        return 100 * (hmx / baseline - 1), baseline, hmx

    rng = random.Random(4416)
    bootstrap = sorted(estimate({k: [rng.choice(v) for _ in v]
                                for k, v in samples.items()})[0]
                       for _ in range(10000))
    overhead, baseline, hmx = estimate(samples)
    return {
        "files": [str(p) for p in paths],
        "n_per_weight_bits": {k: len(v) for k, v in samples.items()},
        "baseline_sum_us": baseline,
        "hmx_sum_us": hmx,
        "overhead_pct": overhead,
        "paired_bootstrap_percentile_95": [bootstrap[249], bootstrap[9749]],
        "bootstrap_note": "Observed runs only; does not model other routing, "
                          "thermal conditions, full-model TTFT or model quality.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--pool-ffn", action="store_true", help="Pool 1024-token RMSNorm runs of both weight formats")
    args = parser.parse_args()
    result = pool_ffn(args.csv) if args.pool_ffn else [summarize(path) for path in args.csv]
    print(json.dumps(result, indent=2))
