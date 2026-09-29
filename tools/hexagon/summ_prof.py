#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file    summ_prof.py
# @brief   Summarize SIM_PROF logs from run_sim_test.sh profile runs
# @author  dlwlzzero <dlwlzzero@gmail.com>
"""Summarize SIM_PROF logs from run_sim_test.sh profile runs.

Usage: python3 tools/hexagon/summ_prof.py logs/hexagon/sim_prof_*.log
           [--layers 28] [--vocab 151936] [--ghz 2.09] [--ops N]
           [--barrier-cyc 4332]

Prints (1) the raw per-kind pcycles of each log, (2) a table rescaled to
the full model - layer op kinds x (layers / model layers),
MATMUL_LOGITS x (vocab / model vocab); EMBED and the final RMSNORM are
counted once per forward (EMBED is a gather, its cost is
O(n_tokens x hidden) and vocab-independent) - with sim-derived ms per
token at the given clock, and (3) the per-thread instruction split the
simulator prints at exit. All numbers are simulator cycles, not device
time.

Known approximation: RMSNORM is a layer kind, so the x(layers/model
layers) factor also multiplies the single final-norm call instead of
counting it once; at 28 layers that overstates RMSNORM by ~1/57 of its
own share, i.e. ~0.1 % of the total. Percentages are over the scaled op
total excluding the barrier; "ms/tok" includes the barrier.

Caveat: simulator Insns cover the whole run (setup, weight fill, scalar
reference, unmeasured fill chunks) and count DMA-queue busy-wait
instructions; the worker-thread spread reported in the thread-balance
section is an upper bound on kernel imbalance, not a measurement of the
profiled window.

Also prints (4) a per-(kind,k,n) table from the SIM_PROF op= lines when
present.
"""
import argparse
import re
import statistics
import sys

LAYER_KINDS = ["RMSNORM", "MATMUL_W8A8", "ROPE", "ATTN", "SILU_MUL", "ADD",
               "MATMUL_W8A16", "MATMUL_W4A8"]
VOCAB_KINDS = ["MATMUL_LOGITS"]
KINDS = ["EMBED", "RMSNORM", "MATMUL_W8A8", "ROPE", "ATTN", "SILU_MUL", "ADD",
         "MATMUL_LOGITS", "MATMUL_W8A16", "MATMUL_W4A8"]


def parse(path):
    r = {"path": path, "kinds": {}, "threads": [], "timing": "off", "ops": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            m = re.match(r"SIM_PROF model (.*)", ln)
            if m:
                r.update({k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))})
            m = re.match(r"SIM_PROF scenario=(\S+) workers=(\d+) tokens=(\d+) pos=(\d+) total_pcycles=(\d+)", ln)
            if m:
                r["scenario"], r["workers"], r["tokens"], r["pos"], r["total"] = (
                    m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
            m = re.match(r"SIM_PROF kind=(\S+) calls=(\d+) pcycles=(\d+) per_call=(\d+)", ln)
            if m:
                if m.group(1) not in KINDS:
                    sys.exit(f"{path}: unknown SIM_PROF kind={m.group(1)}; "
                             f"update KINDS/LAYER_KINDS/VOCAB_KINDS in "
                             f"{__file__} to say how it rescales")
                r["kinds"][m.group(1)] = (int(m.group(2)), int(m.group(3)))
            m = re.match(r"SIM_PROF op=(\d+) kind=(\S+) layer=(\d+) k=(\d+) n=(\d+) pcycles=(\d+)", ln)
            if m:
                r["ops"].append((int(m.group(1)), m.group(2), int(m.group(3)),
                                 int(m.group(4)), int(m.group(5)), int(m.group(6))))
            m = re.match(r"SIM_PROF barrier_empty_x1000=(\d+)", ln)
            if m:
                r["barrier"] = int(m.group(1)) / 1000.0
            m = re.match(r"T(\d+): Insns=(\d+) Packets=(\d+)", ln)
            if m:
                r["threads"].append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            m = re.match(r"SIM_RUN timing=(on|off)", ln)
            if m:
                r["timing"] = m.group(1)
            # Kept for logs that echo the sim command line; the SIM_RUN
            # timing= marker from run_sim_test.sh is the primary source.
            if "--timing" in ln:
                r["timing"] = "on"
            if ln == "SIM_TEST profile PASS":
                r["pass"] = True
    if "scenario" not in r or not r.get("pass"):
        sys.exit(f"{path}: no complete SIM_PROF run (missing scenario or PASS)")
    return r


def scaled(r, layers, vocab, ghz, ops, barrier_cyc=None):
    """Rescale one run to the full model.

    Percentages printed by the caller are over the scaled op total
    excluding the barrier; the returned ms/tok includes it. Kinds not in
    LAYER_KINDS or VOCAB_KINDS (EMBED) scale by 1: one call per forward.
    """
    lf = layers / r["layers"]
    vf = vocab / r["vocab"]
    out = {}
    for k in KINDS:
        cyc = r["kinds"].get(k, (0, 0))[1]
        f = lf if k in LAYER_KINDS else (vf if k in VOCAB_KINDS else 1.0)
        out[k] = cyc * f
    total = sum(out.values())
    bc = barrier_cyc if barrier_cyc is not None else r.get("barrier")
    barrier = bc * ops if bc is not None else None
    ms_per_tok = (total + (barrier or 0.0)) / (ghz * 1e9) * 1e3 / r["tokens"]
    return out, total, barrier, ms_per_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--ghz", type=float, default=2.09)
    ap.add_argument("--ops", type=int, default=None,
                    help="ops per forward (default 1 + 16*layers + 2: embed, "
                         "16 ops per layer, final norm + logits)")
    ap.add_argument("--barrier-cyc", type=float, default=None,
                    help="override the per-op barrier cost for every row so "
                         "all ms/tok share one definition")
    a = ap.parse_args()
    if a.ops is None:
        a.ops = 1 + 16 * a.layers + 2
    runs = [parse(p) for p in a.logs]

    print("## raw (sim pcycles, model as run)\n")
    print("| scenario | workers | timing | tokens | total | " + " | ".join(KINDS) + " | barrier/op |")
    print("|---|---|---|---|---|" + "---|" * len(KINDS) + "---|")
    for r in runs:
        cells = [str(r["kinds"].get(k, (0, 0))[1]) for k in KINDS]
        barrier_cell = f"{r['barrier']:.0f}" if "barrier" in r else "n/a"
        print(f"| {r['scenario']} | {r['workers']} | {r['timing']} | {r['tokens']} | {r['total']} | "
              + " | ".join(cells) + f" | {barrier_cell} |")

    ovr = (f", barrier/op override = {a.barrier_cyc:g} cyc"
           if a.barrier_cyc is not None else "")
    print(f"\n## scaled to {a.layers} layers / vocab {a.vocab} (sim-derived ms at {a.ghz} GHz, NOT device time; {a.ops} ops/token{ovr})\n")
    print("| scenario | workers | timing | ms/tok | barrier ms/tok | " + " | ".join(KINDS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(KINDS))
    have_barrier = []
    for r in runs:
        out, total, barrier, ms = scaled(r, a.layers, a.vocab, a.ghz, a.ops,
                                         a.barrier_cyc)
        have_barrier.append(barrier is not None)
        bms = f"{barrier / (a.ghz * 1e9) * 1e3 / r['tokens']:.3f}" if barrier is not None else "n/a"
        cells = [f"{out[k] / total * 100:.2f}%" for k in KINDS]
        print(f"| {r['scenario']} | {r['workers']} | {r['timing']} | {ms:.2f} | {bms} | " + " | ".join(cells) + " |")
    if a.barrier_cyc is None and any(have_barrier) and not all(have_barrier):
        print("\nWarning: some rows include the fork-join barrier in ms/tok "
              "and others do not (their log has no barrier_empty_x1000 line); "
              "pass --barrier-cyc to make every row comparable.")

    print("\n## thread balance (simulator Insns per hardware thread)\n")
    print("Caveat: simulator Insns cover the whole run (setup, weight fill, scalar "
          "reference, unmeasured fill chunks) and count DMA-queue busy-wait "
          "instructions; the spread over T1..T<n> is an upper bound on kernel "
          "imbalance, not a measurement of the profiled window.\n")
    for r in runs:
        if not r["threads"]:
            continue
        workers = r["workers"]
        worker_ins = [t[1] for t in r["threads"] if 1 <= t[0] <= workers]
        if len(worker_ins) >= 2:
            spread = f"{(max(worker_ins) - min(worker_ins)) / statistics.mean(worker_ins) * 100:.0f}%"
        else:
            spread = "n/a"
        labels = [f"T{t[0]}(main)={t[1]}" if t[0] == 0 else f"T{t[0]}={t[1]}" for t in r["threads"]]
        print(f"- {r['scenario']} w{r['workers']}: " + ", ".join(labels)
              + f" (worker spread T1..T{workers} {spread})")

    print("\n## per-shape (raw pcycles, model as run; ops grouped by kind,k,n)\n")
    for r in runs:
        if not r["ops"]:
            print(f"- {r['scenario']} w{r['workers']}: no SIM_PROF op= lines")
            continue
        buckets = {}
        for _, kind, _, k, n, cyc in r["ops"]:
            calls, tot = buckets.get((kind, k, n), (0, 0))
            buckets[(kind, k, n)] = (calls + 1, tot + cyc)
        print(f"### {r['scenario']} w{r['workers']}\n")
        print("| kind | k | n | calls | pcycles | per_call |")
        print("|---|---|---|---|---|---|")
        for (kind, k, n), (calls, tot) in sorted(buckets.items(), key=lambda x: -x[1][1]):
            print(f"| {kind} | {k} | {n} | {calls} | {tot} | {tot // calls} |")


if __name__ == "__main__":
    main()
