#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package htp_fc_report
# @brief Render the HTP fully-connected device run as charts in the terminal.
"""Render the HTP fully-connected device run as charts in the terminal.

The counterpart to tools/htp_attn_report.py, for the other half of the QNN
comparison table. Reads the ``FC_FIELD`` / ``FC_STAGE`` marker lines
unittest_hvx_fc.cpp emits and draws them. Markers are keyed by name, never by
column position: a position-keyed report script silently printed a stale number
for every shape past the first in this project, twice (MHA_HTP_PLAN.md 9.7).

Usage
    tools/htp_fc_report.py                          # /tmp/hvx_fc_device_run.log
    tools/htp_fc_report.py run.log ...              # or files you name
    tools/htp_fc_report.py --qnn-profile            # vs QNN stage by stage
    tools/htp_fc_report.py --qnn 512=1.45,1024=3.21 # overlay QNN ms per M
                                                    # (those two are the
                                                    #  ATTENTION numbers --
                                                    #  pass the FC ones)
    tools/htp_fc_report.py --html report.html       # also write a HTML report
    tools/htp_fc_report.py --no-color               # plain text

Stdlib only: this runs on whatever machine the device is plugged into.
"""

import argparse
import html
import os
import re
import shutil
import sys

DEFAULT_LOGS = ["/tmp/hvx_fc_device_run.log"]

MARKER = re.compile(
    r"^(?P<kind>FC_FIELD|FC_STAGE)\s+path=(?P<path>\S+)\s+"
    r"field=(?P<field>\S+)\s+value=(?P<value>\S+)\s*$"
)

# The five probes subdivide the DSP-internal time. What is left after all five
# is the micro-mms plus loop overhead, which the two-parameter cost model
# (31_dataflow_as_built.md 3) predicts at ~0.38 us per 64x32x32 micro-mm --
# so the remainder is not a leftover, it is the term with a prediction.
PROBES = ["quant_us", "dequant_us", "acc_read_us", "acc_copy_us", "drain_us"]
PROBE_LABEL = {
    "quant_us": "quant     (act f32 -> u8 AH, VTCM)",
    "dequant_us": "dequant   (i32 -> f32, in place)",
    "acc_read_us": "acc_read  (HMX acc -> VTCM, vendor)",
    "acc_copy_us": "acc_copy  (VTCM -> DDR staging)",
    "drain_us": "drain     (DMA waits)",
}
PROBE_COLOR = {"quant_us": 3, "dequant_us": 2, "acc_read_us": 4,
               "acc_copy_us": 1, "drain_us": 5}
PROBE_GLYPH = {"quant_us": "=", "dequant_us": "+", "acc_read_us": "#",
               "acc_copy_us": "*", "drain_us": "."}
PROBE_HTML = {"quant_us": "#e6a817", "dequant_us": "#54a24b",
              "acc_read_us": "#4c78a8", "acc_copy_us": "#e45756",
              "drain_us": "#b279a2"}
REMAINDER_HTML = "#8c8c8c"

# QNN's own per-stage breakdown of one fully connected op, recorded here so the
# comparison is against numbers that are written down rather than remembered.
# Reported by the QNN side of this evaluation; the shape is q_proj and the
# sequence length is not stated -- qnn_implied_m() below backs it out.
QNN_PROFILES = {
    "u8i4": {
        "label": "QNN u8i4, one FC op (reported on S26 Ultra)",
        "init_us": 5097.0,        # context/binary load, once
        "cold_us": 1263.0,        # first inference
        "netrun_us": 513.0,       # execute() entry to return
        "host_overhead_us": 128.0,  # NetRun - RPC execute
        "rpc_execute_us": 385.0,  # host <-> device
        "non_accel_rpc_us": 39.0,  # RPC execute - accelerate execute
        "accel_execute_us": 347.0,  # the device timeline
        "input_us": 44.5,         # input slice / load (uint8)
        "fc_us": 69.9,            # FC: weight load + compute
        "outfmt_us": 151.1,       # output format / layout (uint8)
        "writeback_us": 72.9,     # output writeback
        "idle_us": 5.2,           # idle / misc / overlap
    },
}

# What lines up with what. QNN's graph is uint8 in and uint8 out, so it has no
# stage that corresponds to our f32 quant/dequant -- their "output format" is
# the accumulator drain plus a requantize, which is the same work our acc_read
# and dequant do, at a quarter of the output bytes. Our terms are given as a
# list of FC_STAGE fields to add up; "__compute" is dsp_total minus every
# probe that is data movement, i.e. the micro-mms and the weight DMA.
QNN_MAP = [
    ("Input slice / load (uint8)", "input_us",
     ["quant_us"], "quant (f32 -> u8 AH)"),
    ("FC (weight load + compute)", "fc_us",
     ["__compute"], "micro-mm + weight DMA"),
    ("Output format / layout (uint8)", "outfmt_us",
     ["acc_read_us", "dequant_us"], "acc_read + dequant"),
    ("Output writeback", "writeback_us",
     ["acc_copy_us"], "acc_copy (0 when in place)"),
    ("Idle / misc / overlap", "idle_us", [], "(unattributed)"),
]

# HMX int8 micro-mm geometry, mirrored from hexkl_micro.h, and the per-mm cost
# the cost model measured on this device.
TILE_ROW, TILE_INNER, TILE_COL = 64, 32, 32
C_MM_US = 0.38

BLOCKS = " ▏▎▍▌▋▊▉█"


class Ink:
    """ANSI colour, off when the output is not a terminal."""

    def __init__(self, on):
        self.on = on

    def __call__(self, s, code=None, bold=False):
        if not self.on or code is None and not bold:
            return s
        pre = ""
        if bold:
            pre += "\033[1m"
        if code is not None:
            pre += f"\033[3{code}m"
        return f"{pre}{s}\033[0m"

    def dim(self, s):
        return f"\033[2m{s}\033[0m" if self.on else s


def parse(paths):
    out = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing {p})", file=sys.stderr)
            continue
        with open(p, errors="replace") as fh:
            for line in fh:
                m = MARKER.match(line.strip())
                if not m:
                    continue
                try:
                    v = float(m["value"])
                except ValueError:
                    v = m["value"]
                out[(m["kind"], m["path"], m["field"])] = v
    return out


def get(rec, kind, path, field, default=None):
    return rec.get((kind, path, field), default)


def nh_of(rec, path):
    """Handles per call, i.e. how many matmuls one call carried.

    A group call amortises everything that happens ONCE PER CALL -- the first
    weight's synchronous DMA drain, and the activation quantization it shares
    -- across n_handles matmuls. So every per-call number has to be divided by
    this before it can sit beside a one-handle row. Comparing the totals is
    what made the cross path read as a regression when it is a 1.2-1.5x win.
    See kGroups in test/unittest/unittest_hvx_fc.cpp.
    """
    return int(get(rec, "FC_FIELD", path, "n_handles", 1) or 1)


def rows(rec):
    """Every (path, shape name, M, K, N) the log carries, in M order."""
    out = []
    for (kind, path, field) in rec:
        if kind != "FC_FIELD" or field != "M" or not path.startswith("fc_"):
            continue
        m = re.match(r"fc_(.+)_m(\d+)$", path)
        if not m:
            continue
        K = get(rec, "FC_FIELD", path, "K")
        N = get(rec, "FC_FIELD", path, "N")
        if K is None or N is None:
            continue
        out.append((path, m[1], int(m[2]), int(K), int(N)))
    return sorted(set(out), key=lambda r: (r[1], r[2]))


def bar(value, vmax, width, glyph=None, pad=False):
    """A horizontal bar; a repeated glyph when it has to survive without
    colour, Unicode eighths otherwise. @a pad fills the cell BEFORE any colour
    escape is wrapped around it, or the padding counts invisible bytes."""
    if vmax <= 0:
        return " " * width if pad else ""
    n = max(0.0, min(1.0, value / vmax)) * width
    if glyph:
        out = glyph * int(round(n))
    else:
        full = int(n)
        rest = int((n - full) * 8)
        out = "█" * full + (BLOCKS[rest] if rest else "")
    return out.ljust(width) if pad else out


def rule(ink, title, width):
    head = f"─ {title} "
    return ink("╭" + head + "─" * max(0, width - len(head) - 1), 6, bold=True)


def header(ink, width):
    return [ink("  HTP fully connected layer -- device run", 6, bold=True),
            ink.dim("  uint8 activation x int8 weight. avg/min/max are over "
                    "ALL 10 iterations,"),
            ink.dim("  run 1 included -- the convention the QNN numbers are "
                    "quoted under."),
            ""]


def env(rec, ink, width):
    q = get(rec, "FC_FIELD", "env", "rpc_poll_qos")
    io = get(rec, "FC_FIELD", "env", "ion_buffers")
    if q is None and io is None:
        return []
    qtxt = {2: ("poll mode", 2), 1: ("PM mode", 3),
            0: ("REJECTED at runtime", 1)}.get(
        int(q) if q is not None else -1, ("not reported", 1))
    itxt = {1: ("ION-backed", 2),
            0: ("unavailable (plain heap)", 1)}.get(
        int(io) if io is not None else -1, ("not reported", 1))
    out = [rule(ink, "Transport controls this run actually had", width),
           "  RPC latency QoS   " + ink(qtxt[0], qtxt[1], bold=True),
           "  act/out buffers   " + ink(itxt[0], itxt[1], bold=True)]
    if (q is not None and q <= 0) or (io is not None and io <= 0):
        out.append(ink.dim("  transport numbers below were measured WITHOUT "
                           "the disabled controls."))
    return out + [""]


def gates(rec, ink, width):
    rs = rows(rec)
    if not rs:
        return []
    out = [rule(ink, "Gate", width)]
    for path, name, M, _K, _N in rs:
        v = get(rec, "FC_FIELD", path, "timed_matches_production")
        label = f"{name} M={M}: timed == production, bitwise"
        if v is None:
            out.append(f"  {ink('?', 3)} {label:<44} "
                       + ink.dim("not reported (pre-timed-endpoint build)"))
            continue
        mark = (ink("PASS", 2, bold=True) if v == 1
                else ink("FAIL", 1, bold=True))
        out.append(f"  {mark} {label:<44} "
                   + ink.dim("else the breakdown describes another kernel"))
    return out + [""]


def macs(M, K, N):
    return float(M) * K * N


def micro_mms(M, K, N):
    """Micro-mms hexkl_mm_u8i8_layer_run issues: one per (row block, N tile,
    K tile), with M rounded up to the 64-row accumulator."""
    m_pad = -(-M // TILE_ROW) * TILE_ROW
    return (m_pad // TILE_ROW) * (N // TILE_COL) * (K // TILE_INNER)


def headline(rec, ink, width, qnn):
    out = [rule(ink, "Cost per layer, us -- wall (with FastRPC) and dsp "
                     "(without)", width)]
    rs = rows(rec)
    if not rs:
        return out + ["  (no FC timings in these logs)", ""]

    def per_mm(path, kind, field):
        v = get(rec, kind, path, field)
        return None if v is None else v / nh_of(rec, path)

    if any(nh_of(rec, p) > 1 for p, _n, _m, _k, _N in rs):
        out.append(ink.dim("  PER MATMUL: a group row's call total is divided "
                           "by its handle count, so it can be read"))
        out.append(ink.dim("  against the one-handle row above it. The call "
                           "total is on the iters line."))
        out.append("")
    vmax = max(per_mm(p, "FC_FIELD", "us_avg_1_10") or 0.0
               for p, _n, _m, _k, _N in rs)
    barw = max(16, width - 64)
    lw = 28  # wide enough for "qproj_i4_x3 1024x2048 M=1024"
    for path, name, M, K, N in rs:
        nh = nh_of(rec, path)
        avg = per_mm(path, "FC_FIELD", "us_avg_1_10")
        if avg is None:
            continue
        lo = per_mm(path, "FC_FIELD", "us_min")
        hi = per_mm(path, "FC_FIELD", "us_max")
        first = per_mm(path, "FC_FIELD", "us_first")
        ini = get(rec, "FC_FIELD", path, "init_us")
        dsp = per_mm(path, "FC_STAGE", "dsp_total_us")
        tr = per_mm(path, "FC_STAGE", "transport_us")
        label = f"{name} {K}x{N} M={M}"
        out.append(f"  {label:<{lw}}{ink(bar(avg, vmax, barw, pad=True), 4)}"
                   f"{avg:>10,.0f}")
        line = f"  {'':<{lw}}10 iters: avg(1-10) {avg:,.0f}"
        if lo is not None and hi is not None:
            line += f"  min {lo:,.0f}  max {hi:,.0f}"
        if first is not None:
            line += f"  run1 {first:,.0f}"
        if ini is not None:
            line += f"  |  init {ini:,.0f}"
        if nh > 1:
            line += f"  |  {nh} matmuls/call, call total {avg * nh:,.0f}"
        out.append(ink.dim(line))
        # dsp and transport BOTH come from the instrumented run, so they add
        # up to each other and not to the production wall above. Subtracting
        # dsp from that wall would charge FastRPC for the instrumentation.
        if dsp and tr is not None:
            out.append(ink.dim(
                f"  {'':<{lw}}probed run: dsp {dsp:,.0f} + FastRPC {tr:,.0f} "
                f"= {dsp + tr:,.0f} us  "
                f"({tr / (dsp + tr) * 100:.0f}% transport)"))
        elif dsp:
            out.append(ink.dim(f"  {'':<{lw}}dsp {dsp:,.0f} us"))
        q = qnn.get(M)
        if q is not None and dsp:
            # QNN quotes a per-op number with no host transfer in it, so the
            # honest term to put beside it is dsp, not wall. Both are shown
            # so the reader can see which comparison they are reading.
            r_dsp = dsp / 1000.0 / q
            r_wall = avg / 1000.0 / q
            verdict = (f"{r_dsp:,.2f}x QNN on dsp" if r_dsp >= 1.0
                       else f"{1.0 / r_dsp:,.2f}x FASTER than QNN on dsp")
            out.append("  " + ink(f"{'':<{lw - 2}}QNN {q:,.3f} ms  ->  "
                                  f"{verdict}   ({r_wall:,.2f}x on wall)", 3))
        out.append("")
    return out


def efficiency(rec, ink, width):
    """Is the number bandwidth-bound, issue-bound, or neither?

    At M=1 a fully connected layer reads the entire weight to do K*N MACs, so
    it is a memory test wearing a matmul's clothes; at M=1024 it is 32,768
    micro-mms and the cost model has a prediction for exactly that. Printing
    both terms is what keeps a slow number from being read as a slow kernel
    when it is really a full pipe.
    """
    rs = rows(rec)
    if not rs:
        return []
    out = [rule(ink, "What the number is bound by", width),
           ink.dim("  weight bytes are read once per call (i8, K*N); payload "
                   "is M*(K+N) floats over FastRPC"),
           ""]
    out.append(f"  {'shape':<20}{'GMAC/s':>9}{'micro-mm':>10}"
               f"{'model us':>10}{'measured':>10}{'payload':>10}"
               f"{'GB/s':>8}")
    padded = []
    for path, name, M, K, N in rs:
        dsp = get(rec, "FC_STAGE", path, "dsp_total_us")
        avg = get(rec, "FC_FIELD", path, "us_avg_1_10")
        if not dsp or not avg:
            continue
        nh = nh_of(rec, path)
        mm = micro_mms(M, K, N)
        model_us = mm * C_MM_US
        # The remainder after the five probes is what the model predicts; the
        # probes are the terms it does not model. Both sides are per matmul --
        # micro-mm count is per matmul, so dsp has to be divided to match, or
        # a group row reads as n_handles times slower than the model.
        probe_sum = sum(get(rec, "FC_STAGE", path, pr) or 0.0 for pr in PROBES)
        measured_mm = (dsp - probe_sum) / nh
        gmacs = macs(M, K, N) / (dsp / nh) / 1000.0
        # One activation in, n_handles output blocks out.
        payload_mb = M * (K + nh * N) * 4.0 / 1048576.0
        transport_us = get(rec, "FC_STAGE", path, "transport_us")
        gbs = (payload_mb / 1024.0) / (transport_us / 1e6) if transport_us \
            else 0.0
        m_pad = -(-M // TILE_ROW) * TILE_ROW
        if m_pad != M:
            padded.append((name, M, m_pad))
        out.append(f"  {name + ' M=' + str(M):<20}{gmacs:>9,.1f}{mm:>10,}"
                   f"{model_us:>10,.0f}{measured_mm:>10,.0f}"
                   f"{payload_mb:>9,.2f}M{gbs:>8,.2f}")
    out.append("")
    out.append(ink.dim(f"  model us = {C_MM_US} us x micro-mm count "
                       f"(31_dataflow_as_built.md 3); measured is dsp_total "
                       f"minus all five probes."))
    out.append(ink.dim("  GB/s is the payload over the measured transport, "
                       "i.e. what FastRPC actually achieved."))
    for name, M, m_pad in padded:
        # GMAC/s counts the MACs the CALLER asked for. The accumulator is 64
        # rows wide, so a shorter M still issues a full block -- at M=1 that
        # is 64x the work for 1x the answer, and reading the low GMAC/s as a
        # slow kernel would be reading the padding.
        out.append(ink.dim(
            f"  {name} M={M}: the 64-row accumulator makes the hardware do "
            f"M={m_pad}, so GMAC/s counts {m_pad // M if M else 0}x fewer "
            f"MACs than were issued."))
    return out + [""]


def breakdown(rec, ink, width):
    out = [rule(ink, "Where the DSP time goes", width)]
    legend = "  ".join(ink(PROBE_GLYPH[p] + " " + p.replace("_us", ""),
                           PROBE_COLOR[p]) for p in PROBES)
    out += ["  " + legend + "  " + ink.dim("_ micro-mm (remainder)"), ""]

    drew = False
    for path, name, M, K, N in rows(rec):
        dsp = get(rec, "FC_STAGE", path, "dsp_total_us")
        if not dsp:
            continue
        drew = True
        avg = get(rec, "FC_FIELD", path, "us_avg_1_10")
        transport = get(rec, "FC_STAGE", path, "transport_us")
        stride = get(rec, "FC_STAGE", path, "acc_stride")

        nh = nh_of(rec, path)
        title = f"{name} M={M}"
        line = f"  {ink(title, None, bold=True)}   dsp {dsp:,.0f} us"
        if transport:
            line += (f" + transport {transport:,.0f} us"
                     f" = probed {dsp + transport:,.0f} us")
        out.append(line)
        if avg:
            out.append(ink.dim(f"    production wall (no instrumentation) "
                               f"{avg:,.0f} us"))

        barw = max(20, width - (70 if nh > 1 else 58))
        if nh > 1:
            out.append(ink.dim(f"    every value is the whole call, for "
                               f"{nh} matmuls; us/mm is the amortised share"))
        def cell(v):
            """us and % of the call, then the per-matmul share for a group."""
            t = f"{v:>9,.0f} us{v / dsp * 100:>7.1f}%"
            return t + (f"{v / nh:>9,.1f}/mm" if nh > 1 else "")
        for pr in PROBES:
            v = get(rec, "FC_STAGE", path, pr) or 0.0
            out.append(f"    {PROBE_LABEL[pr]:<36}"
                       + ink(bar(v, dsp, barw, PROBE_GLYPH[pr], pad=True),
                             PROBE_COLOR[pr]) + cell(v))
        rest = dsp - sum(get(rec, "FC_STAGE", path, pr) or 0.0 for pr in PROBES)
        out.append(ink.dim(f"    {'micro-mm + loop (remainder)':<36}"
                           + bar(rest, dsp, barw, "_", pad=True) + cell(rest)))
        if stride is not None:
            state = (f"in-place tile dequant ACTIVE (row stride {stride:.0f})"
                     if stride else
                     "FELL BACK to the vendor copy -- acc_copy is the DDR "
                     "round trip")
            out.append(ink.dim(f"    {state}"))
        out.append("")
    if not drew:
        return out + ["  (no FC_STAGE fields -- needs a skel built after the "
                      "mm_u8i8_layer_timed commit)", ""]
    return out


def qnn_implied_m(prof, K, N):
    """Which M could QNN's fc_us possibly have been measured at?

    Their table does not say, and it changes what the number means. Two facts
    bound it: the MACs a given M needs, and the weight bytes the op must read
    at least once. Print both rates per candidate M and let the impossible
    ones disqualify themselves -- an implied MAC rate in the thousands of
    GMAC/s is not a fast implementation, it is the wrong M.
    """
    out = []
    w_bytes = K * N / 2.0  # u8i4: two weight values per byte
    for M in (1, 512, 1024):
        gmacs = (float(M) * K * N) / prof["fc_us"] / 1000.0
        out.append((M, gmacs))
    return out, w_bytes / prof["fc_us"] / 1000.0


def qnn_compare(rec, ink, width, prof_name, at_m):
    """QNN's stage breakdown beside ours, one column per weight width.

    The point of this section is NOT the ratio. It is that both stacks split
    an FC op the same way -- a small multiply surrounded by a large amount of
    getting data into and out of the accumulator -- and QNN's own numbers say
    so before ours are even in: their output-format bucket is 2.2x their
    compute bucket. A comparison that only quoted totals would hide that the
    two stacks are losing to the same thing.

    Both widths get a column because only one of them is a like-for-like
    comparison -- QNN's weight is i4 -- while the other is what this tree
    shipped first, and reading either alone has misled once already.
    """
    prof = QNN_PROFILES.get(prof_name)
    if prof is None:
        return []
    out = [rule(ink, f"Against QNN's own breakdown -- {prof['label']}", width)]

    # One column per width, isolated (single handle) rows only: QNN quotes an
    # isolated op, so that is the row the table proper compares. The group
    # rows are picked up separately for the verdict below.
    at = [r for r in rows(rec) if r[2] == at_m] or rows(rec)
    if at and at[0][2] != at_m:
        out.append(ink.dim(f"  (no M={at_m} row in the log; comparing against "
                           f"M={at[0][2]})"))
        at_m = at[0][2]

    def pick(tag, grouped):
        """The path for one width, isolated or grouped. tag None means an
        untagged log from before the width sweep existed."""
        for path, name, M, K, N in at:
            if M != at_m:
                continue
            if (nh_of(rec, path) > 1) != grouped:
                continue
            if tag is None or ("_" + tag) in name:
                return path, K, N
        return None, None, None

    tags = [t for t in ("i4", "i8") if pick(t, False)[0] is not None]
    if not tags:
        tags = [None]  # an older log with no width in the path name
    cols = [(t, pick(t, False)[0]) for t in tags]
    K = N = None
    for t in tags:
        _p, K, N = pick(t, False)
        if K:
            break

    def stage(path, fields):
        """One HexKL cell. __compute is dsp minus everything that only moves
        data, i.e. the term QNN's FC bucket is."""
        if path is None:
            return None
        if fields == ["__compute"]:
            # The test publishes this per probe run; prefer its median over
            # deriving one from the median stage set, and see spread() for why
            # the range matters at small M.
            med = get(rec, "FC_STAGE", path, "compute_us_med")
            if med is not None:
                return med
            dsp = get(rec, "FC_STAGE", path, "dsp_total_us")
            if not dsp:
                return None
            moved = sum(get(rec, "FC_STAGE", path, f) or 0.0
                        for f in ("quant_us", "dequant_us", "acc_read_us",
                                  "acc_copy_us"))
            return dsp - moved
        if not fields:
            return None
        vals = [get(rec, "FC_STAGE", path, f) for f in fields]
        if all(v is None for v in vals):
            return None
        return sum(v or 0.0 for v in vals)

    LW, QW, OW, CW = 30, 8, 26, 9
    SEP = "  " + "-" * (LW + QW) + "|" + "-" * (OW + CW * len(cols) + 4)

    def line(qlabel, qv, olabel, vals):
        lt = f"{qv:>{QW},.1f}" if qv is not None else f"{'-':>{QW}}"
        cells = "".join(f"{v:>{CW},.0f}" if v is not None else f"{'-':>{CW}}"
                        for v in vals)
        return f"  {qlabel:<{LW}}{lt}  |  {olabel:<{OW}}{cells}"

    # Which M our columns are, stated on the face of the table: the two sides
    # are not the same row count, and a reader who assumes they are will read
    # our quant and dequant as much worse than they are.
    out.append(ink.dim(
        f"  our columns are M={at_m}; QNN's row count is not stated and backs "
        f"out to M=1 (see below)."))
    if at_m > 1:
        out.append(ink.dim(
            f"  so our quant and dequant cover {at_m} rows against QNN's 1, "
            f"while the accumulator is 64 rows"))
        out.append(ink.dim(
            "  wide either way -- the multiply and the weight DMA are the "
            "row-count-independent terms."))
    out.append("")
    head_cells = "".join(f"{('u8' + t) if t else 'us':>{CW}}" for t, _p in cols)
    out.append(f"  {'QNN':<{LW}}{'us':>{QW}}  |  {'HexKL':<{OW}}{head_cells}")
    out.append(SEP)
    for qlabel, qfield, ofields, olabel in QNN_MAP:
        out.append(line(qlabel, prof[qfield], olabel,
                        [stage(p, ofields) for _t, p in cols]))
    out.append(SEP)
    out.append(line("accelerate execute (device)", prof["accel_execute_us"],
                    "dsp_total",
                    [get(rec, "FC_STAGE", p, "dsp_total_us") for _t, p in cols]))
    out.append(line("NetRun - accelerate execute",
                    prof["netrun_us"] - prof["accel_execute_us"],
                    "transport (FastRPC)",
                    [get(rec, "FC_STAGE", p, "transport_us") for _t, p in cols]))
    out.append(line("NetRun (steady state)", prof["netrun_us"], "wall",
                    [get(rec, "FC_FIELD", p, "us_avg_1_10") for _t, p in cols]))
    out.append(line("one-time init", prof["init_us"], "init (weight bake)",
                    [get(rec, "FC_FIELD", p, "init_us") for _t, p in cols]))
    out.append("")

    # The question this section was added to answer.
    q_compute = prof["fc_us"]
    out.append("  " + ink("COMPUTE ONLY -- no quant, no dequant, no layout:",
                          None, bold=True))

    def spread(path):
        """lo-hi of the compute bucket over the probe runs, or None.

        Published because the probes are 1 us resolution and one acc_read is
        ~0.38 us: at M=1 the split is dominated by sampling, while dsp_total
        is not. A wide range here means read the verdict as an order of
        magnitude, not a ratio."""
        lo = get(rec, "FC_STAGE", path, "compute_us_lo")
        hi = get(rec, "FC_STAGE", path, "compute_us_hi")
        return None if lo is None or hi is None else (lo, hi)

    def verdict_row(label, grouped, colour):
        """One verdict line across every width, isolated or amortised."""
        parts = []
        for t, _p in cols:
            path, _K, _N = pick(t, grouped)
            v = stage(path, ["__compute"])
            if v is None:
                continue
            nh = nh_of(rec, path) if path else 1
            v /= nh
            r = v / q_compute
            txt = (f"{r:,.1f}x QNN" if r >= 1.0
                   else f"{1.0 / r:,.1f}x FASTER")
            name = ("u8" + t) if t else "HexKL"
            sp = spread(path)
            rng = ""
            if sp and (sp[1] - sp[0]) > 0.15 * max(1.0, sp[1]):
                rng = f" [{sp[0] / nh:,.0f}-{sp[1] / nh:,.0f}]"
            parts.append(f"{name} {v:,.0f}{rng} us (" + ink(txt, colour) + ")")
        if not parts:
            return None
        return f"    {label:<22}QNN {q_compute:,.1f} us   " + "   ".join(parts)

    v1 = verdict_row("isolated, 1 handle", False, 3)
    if v1:
        out.append(v1)
    vg = verdict_row("amortised, group", True, 2)
    if vg:
        out.append(vg)
    if not v1 and not vg:
        out.append(ink.dim(f"    QNN {q_compute:,.1f} us   HexKL not measured "
                           f"yet -- run unittest_hvx_fc"))
    q_share = q_compute / prof["accel_execute_us"] * 100.0
    out.append(ink.dim(
        f"    of the device timeline that is multiply: QNN {q_share:.0f}%"))
    out.append(ink.dim(
        "    a bracket is the spread over the probe runs. The probes are 1 us "
        "resolution against a"))
    if at_m == 1:
        out.append(ink.dim(
            "    0.38 us accumulator read, and at M=1 there are only 64 of "
            "them per matmul, so this"))
        out.append(ink.dim(
            "    split is sampling-bound -- read the ratio as indicative. "
            "dsp_total, one timestamp"))
        out.append(ink.dim(
            "    pair per call, is not affected. M=64 is the smallest row "
            "count where the split holds."))
    else:
        out.append(ink.dim(
            f"    0.38 us accumulator read, but at M={at_m} each probe covers "
            f"enough calls for the split to"))
        out.append(ink.dim(
            "    hold: acc_read comes out the same at both widths here, as it "
            "must, since the"))
        out.append(ink.dim(
            "    accumulator readout cannot depend on the weight width."))
    out.append(ink.dim(
        f"    QNN's own output-format bucket is "
        f"{prof['outfmt_us'] / q_compute:.1f}x its compute bucket -- the "
        f"accumulator readout dominates on both stacks."))
    out.append("")

    if K and N:
        cands, gbs = qnn_implied_m(prof, K, N)
        out.append(ink.dim("  QNN's table does not say which M it measured. "
                           "Backed out of fc_us:"))
        for M, gmacs in cands:
            note = ("plausible" if gmacs < 2000.0
                    else "above any HMX int8 peak -- not this M")
            out.append(ink.dim(f"    M={M:<6}{gmacs:>10,.0f} GMAC/s   {note}"))
        out.append(ink.dim(
            f"    and {K * N / 2 / 1048576:.1f} MiB of i4 weight in "
            f"{q_compute:,.1f} us is {gbs:,.1f} GB/s, which is a DDR read: "
            f"read it as M=1 (decode)."))
    out.append("")
    if "i4" in tags:
        out.append(ink.dim(
            "  The u8i4 column is the like-for-like one: QNN's weight is i4. "
            "u8i8 reads twice the"))
        out.append(ink.dim(
            "  weight bytes, and the weight DMA is the only term the width "
            "moves -- which is why the"))
        out.append(ink.dim(
            "  two columns differ by roughly that DMA and by nothing else."))
    out.append(ink.dim(
        "  Not comparable without saying so: QNN's graph is uint8 in AND "
        "uint8 out, so it"))
    out.append(ink.dim(
        "  never pays an f32 quant or dequant, and moves a quarter of the "
        "output bytes we do."))
    out.append(ink.dim(
        "  The rows above are therefore an alignment of PURPOSE, not of "
        "work."))
    return out + [""]


def bar_svg(labels, values, width=760):
    """Stacked probe bars plus the remainder, for the HTML report."""
    if not labels:
        return ""
    keys = PROBES + ["remainder"]
    colors = dict(PROBE_HTML, remainder=REMAINDER_HTML)
    total_max = max(sum(v) for v in values)
    rowh, top = 26, 8
    h = top + rowh * len(labels)
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" height="{h}" '
             f'role="img">']
    for i, (lab, vs) in enumerate(zip(labels, values)):
        y = top + i * rowh
        x = 170.0
        parts.append(f'<text x="0" y="{y + 13}" font-size="12" '
                     f'fill="currentColor">{html.escape(lab)}</text>')
        for k, v in zip(keys, vs):
            w = (width - 170.0) * v / total_max
            if w > 0.3:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" '
                             f'height="16" fill="{colors[k]}">'
                             f'<title>{k}: {v:,.0f} us</title></rect>')
            x += w
    parts.append("</svg>")
    return "".join(parts)


def write_html(rec, path, plain_sections):
    labels, values = [], []
    for p, name, M, _K, _N in rows(rec):
        dsp = get(rec, "FC_STAGE", p, "dsp_total_us")
        if not dsp:
            continue
        vs = [get(rec, "FC_STAGE", p, pr) or 0.0 for pr in PROBES]
        labels.append(f"{name} M={M}")
        values.append(vs + [max(0.0, dsp - sum(vs))])
    legend = " ".join(
        f'<span style="color:{PROBE_HTML[p]}">&#9632;</span> '
        f'{html.escape(p.replace("_us", ""))}' for p in PROBES)
    legend += (f' <span style="color:{REMAINDER_HTML}">&#9632;</span> '
               f'micro-mm')
    body = f"""<title>HTP fully connected device report</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
        margin: 2rem auto; max-width: 940px; padding: 0 1rem; }}
h1 {{ font-size: 1.25rem; }}
pre {{ overflow-x: auto; padding: .75rem; border-radius: 6px;
       background: rgba(127,127,127,.12); }}
</style>
<h1>HTP fully connected layer &mdash; device run</h1>
<p>{legend}</p>
{bar_svg(labels, values)}
<pre>{html.escape(plain_sections)}</pre>
"""
    with open(path, "w") as fh:
        fh.write(body)
    print(f"wrote {path}")


def parse_qnn(s):
    """--qnn 512=1.45,1024=3.21 -> {512: 1.45, 1024: 3.21}, in milliseconds."""
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[int(k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--qnn", metavar="M=ms,...",
                    help="QNN milliseconds per sequence length, to overlay")
    ap.add_argument("--qnn-profile", metavar="NAME", nargs="?",
                    const="u8i4", choices=sorted(QNN_PROFILES),
                    help="compare stage by stage against a recorded QNN "
                         "breakdown (default profile: u8i4)")
    ap.add_argument("--qnn-m", type=int, default=1, metavar="M",
                    help="which of our M rows the QNN profile is compared "
                         "against (default 1 -- see the derivation it prints)")
    ap.add_argument("--html", metavar="FILE")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    use_color = (sys.stdout.isatty() and not args.no_color
                 and os.environ.get("NO_COLOR") is None)
    ink = Ink(use_color)
    cols = min(shutil.get_terminal_size((100, 24)).columns, 110)
    qnn = parse_qnn(args.qnn)

    rec = parse(args.logs or DEFAULT_LOGS)
    if not rec:
        print("no marker lines found -- did the device run produce logs?",
              file=sys.stderr)
        return 1

    out = []
    out += header(ink, cols)
    out += env(rec, ink, cols)
    out += gates(rec, ink, cols)
    out += headline(rec, ink, cols, qnn)
    out += breakdown(rec, ink, cols)
    out += efficiency(rec, ink, cols)
    if args.qnn_profile:
        out += qnn_compare(rec, ink, cols, args.qnn_profile, args.qnn_m)
    print("\n".join(out))

    if args.html:
        plain = Ink(False)
        sections = "\n".join(
            gates(rec, plain, 100) + headline(rec, plain, 100, qnn)
            + breakdown(rec, plain, 100) + efficiency(rec, plain, 100)
            + (qnn_compare(rec, plain, 100, args.qnn_profile, args.qnn_m)
               if args.qnn_profile else []))
        write_html(rec, args.html, sections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
