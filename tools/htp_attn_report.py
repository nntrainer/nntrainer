#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# @package htp_attn_report
# @brief Render the HTP attention device run as charts in the terminal.
"""Render the HTP attention device run as charts in the terminal.

Reads the ``*_FIELD`` / ``ATTN_STAGE`` marker lines the device tests emit and
draws them. The markers are keyed by name, never by column position: a
position-keyed report script silently printed a stale number for every shape
past the first here, twice (MHA_HTP_PLAN.md 9.7).

Usage
    tools/htp_attn_report.py                       # the three default /tmp logs
    tools/htp_attn_report.py run.log ...           # or files you name
    tools/htp_attn_report.py --width I4I4          # breakdown at another width
    tools/htp_attn_report.py --html report.html    # also write a HTML report
    tools/htp_attn_report.py --no-color            # plain text

Stdlib only: this runs on whatever machine the device is plugged into.
"""

import argparse
import html
import os
import re
import shutil
import sys

DEFAULT_LOGS = [
    "/tmp/hvx_mm_u8i4_device_run.log",
    "/tmp/hvx_softmax_device_run.log",
    "/tmp/hvx_attn_device_run.log",
]

MARKER = re.compile(
    r"^(?P<kind>[A-Z0-9_]+_FIELD|ATTN_STAGE)\s+path=(?P<path>\S+)\s+"
    r"field=(?P<field>\S+)\s+value=(?P<value>\S+)\s*$"
)

WIDTHS = ["I8I8", "I4I4", "I8I4", "I4I8"]
# ONE thing varies across these four runs: the width of the Kt cache and the
# width of the V cache. The activation side is u8 in every one of them, so
# spelling it out ("u8i4/u8i8") puts the invariant in the label and buries the
# variable -- which is exactly what made the earlier "I4,I4" read as u4 x i4.
# Write only what changes, and say the K/V ordering once, in the section title.
WIDTH_LABEL = {"I8I8": "i8/i8", "I4I4": "i4/i4",
               "I8I4": "i8/i4", "I4I8": "i4/i8"}
STAGES = ["qk_us", "softmax_us", "pv_us", "accum_us", "gather_us"]
STAGE_LABEL = {
    "qk_us": "Q.Kt",
    "softmax_us": "softmax",
    "pv_us": "P.V",
    "accum_us": "f32 accum",
    "gather_us": "gather+1/l",
}
# One colour per stage, and one glyph each so the stacked bar still reads
# without colour (piped to a file, NO_COLOR, a mono terminal).
STAGE_COLOR = {"qk_us": 4, "softmax_us": 3, "pv_us": 1, "accum_us": 2,
               "gather_us": 5}
STAGE_GLYPH = {"qk_us": "#", "softmax_us": "=", "pv_us": "*", "accum_us": "+",
               "gather_us": "."}
STAGE_HTML = {"qk_us": "#4c78a8", "softmax_us": "#e6a817", "pv_us": "#e45756",
              "accum_us": "#54a24b", "gather_us": "#b279a2"}

# MHA_HTP_PLAN.md 2's predictions, for the gap to be visible rather than
# recomputed by the reader every time.
PREDICTED_US = {"dec": 825.0, "pre": 1250.0}
CPU_BAR_US = {"dec": 1170.0}

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


def shapes(rec):
    seen = []
    for (kind, path, _f) in rec:
        if kind != "ATTN_FIELD" or not path.startswith("forward_"):
            continue
        m = re.match(r"forward_(dec|pre)_kv(\d+)_", path)
        if m and (m[1], int(m[2])) not in seen:
            seen.append((m[1], int(m[2])))
    return sorted(seen, key=lambda t: (t[1], t[0]))


def bar(value, vmax, width, glyph=None, pad=False):
    """A horizontal bar.

    Unicode eighths when drawing solid, a repeated glyph when the bar has to
    survive without colour. @a pad fills the rest of the cell with spaces so
    the column after it lines up -- do that here, before any colour escape is
    wrapped around the result, or the padding counts invisible bytes.
    """
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
    return [ink("  HTP attention -- device run", 6, bold=True),
            ink.dim("  i8/i4 pairs are the Kt and V cache widths, in that "
                    "order. Activations are u8 throughout,"),
            ink.dim("  so i4 means a u8 x i4 matmul -- nothing here is u4."),
            ""]


def env(rec, ink, width):
    q = get(rec, "ATTN_FIELD", "env", "rpc_poll_qos")
    io = get(rec, "ATTN_FIELD", "env", "ion_buffers")
    if q is None and io is None:
        return []
    qtxt = {2: ("poll mode", 2), 1: ("PM mode", 3),
            0: ("REJECTED at runtime", 1),
            -1: ("compiled out (pre-26393ab build)", 1)}.get(
        int(q) if q is not None else -1)
    itxt = {1: ("ION-backed", 2), 0: ("unavailable (plain heap)", 1),
            -1: ("compiled out (pre-26393ab build)", 1)}.get(
        int(io) if io is not None else -1)
    out = [rule(ink, "Transport controls this run actually had", width)]
    out.append("  RPC latency QoS   " + ink(qtxt[0], qtxt[1], bold=True))
    out.append("  q/out buffers     " + ink(itxt[0], itxt[1], bold=True))
    if (q is not None and q <= 0) or (io is not None and io <= 0):
        out.append(ink.dim("  transport numbers below were measured WITHOUT "
                           "the disabled controls."))
    return out + [""]


def gates(rec, ink, width):
    checks = [
        ("S band vs host model, 384 cases", "ATTN_FIELD", "scores",
         "max_rel_err", lambda v: v == 0.0, "bit-identical"),
        ("  its reference dynamic range", "ATTN_FIELD", "scores",
         "min_ref_dynamic_range", lambda v: v > 0, "> 0, else the 0 is vacuous"),
        ("  V width never reaches S", "ATTN_FIELD", "scores",
         "v_width_invariant", lambda v: v == 1, "bitwise invariant"),
        ("  release then re-register", "ATTN_FIELD", "scores",
         "lifecycle_repeatable", lambda v: v == 1, "reproduces bitwise"),
        ("fused forward vs host model", "ATTN_FIELD", "forward",
         "max_rel_err", lambda v: v <= 5e-3, "tolerance 5e-3 at Kt/V = i8/i8"),
        ("blocked softmax, p, 756 cases", "BLOCKED_FIELD", "softmax_blocked",
         "max_abs_err_p", lambda v: v <= 1e-6, "<= 1e-6"),
        ("  l summation-order margin", "BLOCKED_FIELD", "softmax_blocked",
         "l_sum_order_margin", lambda v: v <= 1.0, "<= 1.0"),
    ]
    out = [rule(ink, "Correctness gates", width)]
    for label, kind, path, field, ok, note in checks:
        v = get(rec, kind, path, field)
        if v is None:
            out.append(f"  {ink('?', 3)} {label:<34} {'-':>10}  {ink.dim(note)}")
            continue
        good = ok(v)
        mark = ink("PASS", 2, bold=True) if good else ink("FAIL", 1, bold=True)
        out.append(f"  {mark} {label:<34} {v:>10.3g}  {ink.dim(note)}")
    return out


def per_layer(rec, ink, width):
    out = [rule(ink, "Cost per fused layer, us (wall) -- Kt/V cache width",
                width)]
    rows = []
    for regime, kv in shapes(rec):
        vals = {w: get(rec, "ATTN_FIELD", f"forward_{regime}_kv{kv}_{w}",
                       "us_per_layer") for w in WIDTHS}
        if vals["I8I8"] is None:
            continue
        rows.append((regime, kv, vals))
    if not rows:
        return out + ["  (no forward timing in these logs)"]

    vmax = max(v for _r, _k, vs in rows for v in vs.values() if v)
    barw = max(18, width - 56)
    for regime, kv, vals in rows:
        name = f"kv={kv:<5}{'decode' if regime == 'dec' else 'prefill'}"
        best = vals["I8I8"]
        out.append(f"  {name:<20}{ink(bar(best, vmax, barw, pad=True), 4)}"
                   f"{best:>10,.0f}")
        spread = [f"{WIDTH_LABEL[w]} {vals[w]:,.0f}" for w in WIDTHS if vals[w]]
        out.append(ink.dim(f"  {'':<20}{'  '.join(spread)}"))

        p8 = f"forward_{regime}_kv{kv}_I8I8"
        lo = get(rec, "ATTN_FIELD", p8, "us_min")
        hi = get(rec, "ATTN_FIELD", p8, "us_max")
        first = get(rec, "ATTN_FIELD", p8, "us_first")
        ini = get(rec, "ATTN_FIELD", p8, "init_us")
        if lo is not None and hi is not None:
            out.append(ink.dim(
                f"  {'':<20}10 iters: avg(1-10) {best:,.0f}  min {lo:,.0f}  "
                f"max {hi:,.0f}"
                + (f"  run1 {first:,.0f}" if first is not None else "")
                + (f"  |  init {ini:,.0f}" if ini is not None else "")))

        # us_per_layer is the FORWARD only. A running model also pays this
        # step's KV append, every token, in every layer, before forward can
        # run -- and init above is the whole-cache append, which is not that.
        app = get(rec, "ATTN_FIELD", p8, "append_us")
        step = get(rec, "ATTN_FIELD", p8, "step_us")
        if app is not None and step is not None:
            out.append(ink.dim(
                f"  {'':<20}per step: kv_append {app:,.0f} + forward "
                f"{best:,.0f} = {step:,.0f} us"))

        # Compare the KERNEL against the plan and against CPU, not the wall
        # clock. Wall = dsp + FastRPC transport, and transport is the one term
        # here that is not reproducible (see the note below), so basing a
        # "13x over" on it charges the kernel for measurement noise.
        dsp = get(rec, "ATTN_STAGE", f"forward_{regime}_kv{kv}_I8I8",
                  "dsp_total_us") or best
        pred = PREDICTED_US.get(regime)
        if pred:
            gap = (f"{dsp / pred:,.1f}x over" if dsp > pred
                   else f"UNDER it at {dsp:,.0f} us")
            out.append(ink.dim(
                f"  {'':<20}plan §2 predicts {pred:,.0f} us  ->  "
                f"{gap} (dsp only)"))
        cpu = CPU_BAR_US.get(regime)
        if cpu:
            vs = (f"{dsp / cpu:,.1f}x slower than CPU" if dsp > cpu
                  else f"{cpu / dsp:,.1f}x FASTER than CPU")
            out.append(ink.dim(
                f"  {'':<20}CPU bar {cpu:,.0f} us  ->  {vs} (dsp only)"))
        out.append("")

    # dsp_total repeats to a fraction of a percent; the wall clock does not.
    # Say which of the two the reader should be reasoning about, with the
    # measured spread rather than an adjective.
    for regime, kv in shapes(rec):
        paths = [f"forward_{regime}_kv{kv}_{w}" for w in WIDTHS]
        sd = spread_pct([get(rec, "ATTN_STAGE", p, "dsp_total_us")
                         for p in paths])
        tr = [get(rec, "ATTN_STAGE", p, "transport_us") for p in paths]
        tr = [t for t in tr if t]
        if sd is None or len(tr) < 2:
            continue
        out.append(ink.dim(
            f"  spread over the four widths at kv={kv} "
            f"{'decode' if regime == 'dec' else 'prefill'}: "
            f"dsp {sd:.2f}%, transport {min(tr):,.0f}-{max(tr):,.0f} us"))
    out.append("")
    ovh = [get(rec, "ATTN_STAGE", f"forward_{r}_kv{k}_I8I8",
               "probe_overhead_us") for r, k in shapes(rec)]
    ovh = [v for v in ovh if v is not None]
    if ovh:
        out.append(ink.dim(
            f"  walls above are the UNINSTRUMENTED attn_forward; the probed "
            f"run costs {min(ovh):,.0f}-{max(ovh):,.0f} us more"))
        out.append("")
    tok = get(rec, "ATTN_FIELD", f"forward_dec_kv1024_I8I8",
              "us_per_token_all_layers")
    if tok:
        out.append(f"  {ink('decode, 28 layers:', None, bold=True)} "
                   f"{tok / 1000.0:,.1f} ms of attention per token")
        step = get(rec, "ATTN_FIELD", "forward_dec_kv1024_I8I8", "step_us")
        if step:
            out.append(ink.dim(
                f"  {'':<20}with the per-token KV append: "
                f"{step * 28 / 1000.0:,.1f} ms"))
    return out


def breakdown(rec, ink, width, wd="I8I8"):
    out = [rule(ink, f"Where the DSP time goes (Kt/V = {WIDTH_LABEL[wd]})",
                width)]
    legend = "  ".join(
        ink(STAGE_GLYPH[s] + " " + STAGE_LABEL[s], STAGE_COLOR[s])
        for s in STAGES)
    out += ["  " + legend, ""]

    drew = False
    for regime, kv in shapes(rec):
        path = f"forward_{regime}_kv{kv}_{wd}"
        total = get(rec, "ATTN_STAGE", path, "dsp_total_us")
        if not total:
            continue
        drew = True
        wall = get(rec, "ATTN_FIELD", path, "us_per_layer")
        transport = get(rec, "ATTN_STAGE", path, "transport_us")
        calls = get(rec, "ATTN_STAGE", path, "layer_run_calls")

        parts_sum = sum(get(rec, "ATTN_STAGE", path, st) or 0.0 for st in STAGES)
        title = f"kv={kv} {'decode' if regime == 'dec' else 'prefill'}"
        # dsp_total and transport both come from the INSTRUMENTED run, so
        # they add up to that run's wall, not to us_per_layer -- which is now
        # the uninstrumented production number. Show the equation that
        # balances, and the production wall beside it.
        probed = total + (transport or 0.0)
        out.append(f"  {ink(title, None, bold=True)}"
                   f"   dsp {total:,.0f} us"
                   + (f" + transport {transport:,.0f} us" if transport else "")
                   + f" = probed {probed:,.0f} us"
                   + (f"   [{calls:,.0f} layer_run calls]" if calls else ""))
        if wall:
            out.append(ink.dim(f"    production wall (no instrumentation) "
                               f"{wall:,.0f} us"))

        # The stages partition the DSP-internal time, so they must add up to
        # it. When they do not, the instrumentation is mispaired -- a stale t0
        # spanning several stages inflates one of them -- and drawing a 180%
        # bar would present that as a measurement.
        if parts_sum > total * 1.05:
            out.append("    " + ink(
                f"BROKEN: stages sum to {parts_sum:,.0f} us against a "
                f"{total:,.0f} us total ({100 * parts_sum / total:.0f}%).",
                1, bold=True))
            out.append("    " + ink.dim(
                "The timers are mispaired, not the kernel slow. Not drawing "
                "this."))
            out.append("")
            continue
        if parts_sum < total * 0.8:
            out.append("    " + ink(
                f"note: stages account for {100 * parts_sum / total:.0f}% of "
                f"the DSP total; the rest is unattributed.", 3))

        barw = max(20, width - 50)
        stacked = ""
        for s in STAGES:
            v = get(rec, "ATTN_STAGE", path, s) or 0.0
            seg = bar(v, total, barw, STAGE_GLYPH[s]) if v else ""
            stacked += ink(seg, STAGE_COLOR[s])
        out.append("    " + stacked)

        for s in STAGES:
            v = get(rec, "ATTN_STAGE", path, s)
            if v is None:
                continue
            pct = 100.0 * v / total
            out.append(f"    {STAGE_LABEL[s]:<12}"
                       + ink(bar(v, total, barw, pad=True), STAGE_COLOR[s])
                       + f"{v:>9,.0f} us {pct:>5.1f}%")
        out.append("")
    if not drew:
        out.append("  (no ATTN_STAGE lines -- needs a build with "
                   "attn_forward_timed)")
    return out


# The geometry FusedForwardPerLayerCost runs at (Qwen3-0.6B), repeated here
# because the log carries stage totals rather than the loop bounds that
# produced them. layer_run_calls is the check: if the count derived from these
# does not match what the DSP reported, the constants are stale and no
# per-block number gets published.
ATTN_CFG = {"nch": 8, "gqa": 2, "head_dim": 128, "T": 256, "M_band": 64,
            "n_query_pre": 128}


def geometry(regime, kv):
    """(rows in one band, bands, blocks per head-band, total blocks, calls)."""
    c = ATTN_CFG
    m = (1 if regime == "dec" else c["n_query_pre"]) * c["gqa"]
    bands = -(-m // c["M_band"])
    n_blocks = -(-kv // c["T"])
    return (min(m, c["M_band"]), bands, n_blocks,
            c["nch"] * bands * n_blocks, c["nch"] * bands * (1 + n_blocks))


def spread_pct(vals):
    vals = [v for v in vals if v]
    if len(vals) < 2 or not min(vals):
        return None
    return 100.0 * (max(vals) - min(vals)) / min(vals)


def per_block(rec, ink, width, wd="I8I8"):
    """What one (head, band, block) costs, which is what actually compares.

    us/call across shapes was the old verdict and it was too weak to carry
    one: it mixes the Q.Kt call (which spans every block) with the P.V calls
    (one per block), so it moves with n_blocks for reasons that say nothing
    about the kernel. Normalising each mm stage to one block does compare,
    and it is what makes the M-invariance below readable as a finding.
    """
    out = [rule(ink, f"What one block costs (Kt/V = {WIDTH_LABEL[wd]})", width)]
    rows = []
    for regime, kv in shapes(rec):
        path = f"forward_{regime}_kv{kv}_{wd}"
        calls = get(rec, "ATTN_STAGE", path, "layer_run_calls")
        qk = get(rec, "ATTN_STAGE", path, "qk_us")
        pv = get(rec, "ATTN_STAGE", path, "pv_us")
        sm = get(rec, "ATTN_STAGE", path, "softmax_us")
        if not calls or qk is None or pv is None:
            continue
        mrows, _bands, _nb, blocks, want = geometry(regime, kv)
        if int(calls) != want:
            out.append("  " + ink(
                f"kv={kv} {regime}: {calls:,.0f} layer_run calls, geometry "
                f"says {want}. ATTN_CFG is stale -- not normalising.", 1))
            continue
        rows.append((regime, kv, mrows, qk / blocks, pv / blocks,
                     (sm or 0.0) / blocks))
    if not rows:
        return out + ["  (needs ATTN_STAGE stage totals)"]

    # Padded, deliberately: the integer accumulator is M_band rows wide, so an
    # M=2 band issues exactly the MACs an M=64 one does. Quoting the useful
    # MACs would hide the waste this section exists to show.
    macs = ATTN_CFG["M_band"] * ATTN_CFG["head_dim"] * ATTN_CFG["T"]
    out.append(ink.dim(f"  {'shape':<20}{'rows/band':>10}{'Q.Kt/blk':>10}"
                       f"{'P.V/blk':>10}{'softmax/blk':>13}{'Q.Kt GMAC/s':>13}"))
    for regime, kv, mrows, q, p, s in rows:
        name = f"kv={kv:<5}{'decode' if regime == 'dec' else 'prefill'}"
        out.append(f"  {name:<20}{mrows:>10}{q:>10.1f}{p:>10.1f}{s:>13.2f}"
                   f"{macs / q / 1e3:>13.2f}")
    out.append(ink.dim("  us per (head, band, block); GMAC/s counts the PADDED "
                       "rows the accumulator issues"))
    out.append("")

    # 1. rows: decode and prefill at the same kv differ only in rows/band.
    for _r, kv, mrows, q, p, s in rows:
        if _r != "dec":
            continue
        pre = [r for r in rows if r[0] == "pre" and r[1] == kv]
        if not pre:
            continue
        _r2, _kv2, mrows2, q2, p2, s2 = pre[0]
        out.append(f"  kv={kv}: {mrows2 // mrows}x the rows costs  "
                   + ink(f"Q.Kt x{q2 / q:.2f}", 3) + "  "
                   + ink(f"P.V x{p2 / p:.2f}", 3)
                   + (("  " + ink(f"softmax x{s2 / s:.1f}", 2)) if s else ""))

    # 2. width: the two weight operands, over the same shape.
    for regime, kv in shapes(rec):
        paths = [f"forward_{regime}_kv{kv}_{w}" for w in WIDTHS]
        sq = spread_pct([get(rec, "ATTN_STAGE", p, "qk_us") for p in paths])
        sp = spread_pct([get(rec, "ATTN_STAGE", p, "pv_us") for p in paths])
        if sq is None or sp is None:
            continue
        name = f"kv={kv} {'decode' if regime == 'dec' else 'prefill'}"
        out.append(f"  {name}: all four weight widths lie within  "
                   + ink(f"Q.Kt {sq:.2f}%", 3) + "  " + ink(f"P.V {sp:.2f}%", 3))

    # Derived, not asserted: this exact sentence has gone stale twice as the
    # kernel changed underneath it.
    qk_ratio = None
    dec_rows = [r for r in rows if r[0] == "dec"]
    pre_rows = [r for r in rows if r[0] == "pre" and dec_rows
                and r[1] == dec_rows[0][1]]
    if dec_rows and pre_rows and dec_rows[0][3]:
        qk_ratio = pre_rows[0][3] / dec_rows[0][3]
    if qk_ratio is not None and qk_ratio < 1.6:
        verdict = (
            "The mm stages do not move with the row count: their cost is\n"
            "  per-call overhead, not arithmetic, and narrowing the weights\n"
            "  cannot help until it is gone.")
    else:
        verdict = (
            "The mm stages now scale with the row count -- per-call overhead\n"
            "  no longer dominates them. What remains per call is in the\n"
            "  probes above: quant, the vendor acc_read floor, and the f32\n"
            "  accumulate.")
    out += ["", "  " + ink(verdict, 3)]
    return out


def bar_svg(labels, values, width=760, row_h=26):
    if not values:
        return ""
    total_max = max(sum(v) for v in values) or 1.0
    h = row_h * len(labels) + 20
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" '
             f'style="max-width:{width}px" role="img">']
    for i, (lab, vs) in enumerate(zip(labels, values)):
        y = i * row_h + 4
        x = 150.0
        parts.append(f'<text x="0" y="{y + 13}" font-size="12" '
                     f'fill="currentColor">{html.escape(lab)}</text>')
        for s, v in zip(STAGES, vs):
            w = (width - 170.0) * v / total_max
            if w > 0.3:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" '
                             f'height="16" fill="{STAGE_HTML[s]}">'
                             f'<title>{STAGE_LABEL[s]}: {v:,.0f} us</title>'
                             f'</rect>')
            x += w
    parts.append("</svg>")
    return "".join(parts)


PROBES = ["acc_read_us", "acc_copy_us", "dequant_us", "quant_us", "drain_us"]
PROBE_LABEL = {"acc_read_us": "acc_read  (HMX acc -> VTCM, vendor)",
               "acc_copy_us": "acc_copy  (VTCM -> DDR scratch, ours)",
               "dequant_us": "dequant   (i32 -> f32, DDR -> DDR)",
               "quant_us": "quant     (act f32 -> u8 AH in VTCM)",
               "drain_us": "drain     (DMA waits)"}


def readout_split(rec, ink, width, wd="I8I8"):
    """Tier 0: which half of the 52 us readout is the 52 us.

    The probes subdivide qk_us + pv_us -- same run, same clock -- so the
    remainder after subtracting all five is the micro-mms plus loop overhead,
    and the model predicts THAT at ~0.38 us per mm. acc_read vs acc_copy is
    the decision this section exists for: acc_read is a vendor call whose
    cost we can only avoid (fewer readouts, or no HMX for tiny M), acc_copy
    and dequant are ours to delete (dequant straight from VTCM).
    """
    out = [rule(ink, f"Inside layer_run -- Tier 0 probes (Kt/V = "
                     f"{WIDTH_LABEL[wd]})", width)]
    c = ATTN_CFG
    rows = []
    for regime, kv in shapes(rec):
        path = f"forward_{regime}_kv{kv}_{wd}"
        vals = {pr: get(rec, "ATTN_STAGE", path, pr) for pr in PROBES}
        if vals["acc_read_us"] is None:
            continue
        qkpv = ((get(rec, "ATTN_STAGE", path, "qk_us") or 0.0)
                + (get(rec, "ATTN_STAGE", path, "pv_us") or 0.0))
        _m, bands, nb, _blocks, _calls = geometry(regime, kv)
        readouts = c["nch"] * bands * nb * (c["T"] // 32 + c["head_dim"] // 32)
        rows.append((regime, kv, vals, qkpv, readouts))
    if not rows:
        return out + ["  (no probe fields -- needs a skel built after the "
                      "Tier 0 commit)", ""]

    for regime, kv, vals, qkpv, readouts in rows:
        name = f"kv={kv} {'decode' if regime == 'dec' else 'prefill'}"
        out.append(f"  {ink(name, None, bold=True)}   qk+pv {qkpv:,.0f} us, "
                   f"{readouts} readouts")
        barw = max(20, width - 62)
        for pr in PROBES:
            v = vals[pr] or 0.0
            per = (f"  = {v / readouts:5.1f} us/readout"
                   if pr in ("acc_read_us", "acc_copy_us") else "")
            out.append(f"    {PROBE_LABEL[pr]:<38}"
                       + bar(v, qkpv, barw, pad=True) + f" {v:>9,.0f} us{per}")
        rest = qkpv - sum(vals[pr] or 0.0 for pr in PROBES)
        out.append(ink.dim(f"    {'micro-mm + loop (remainder)':<38}"
                           f"{'':{barw}} {rest:>9,.0f} us"))
        out.append("")

    # The verdict only needs one shape; decode kv=512 is the cleanest (M=2
    # makes quant/dequant negligible, so the two acc legs stand alone).
    dec = [r for r in rows if r[0] == "dec"]
    if dec:
        _r, _kv, vals, _q, _n = dec[0]
        rd, cp = vals["acc_read_us"] or 0.0, vals["acc_copy_us"] or 0.0
        stride = get(rec, "ATTN_STAGE", f"forward_dec_kv{_kv}_{wd}",
                     "acc_stride")
        if stride:
            out += ["  " + ink(
                f"In-place tile dequant is ACTIVE (derived row stride "
                f"{stride:.0f}). The DDR round trip is\n  gone; acc_read is "
                f"the vendor drain floor. The Tier 1 question is settled.",
                2), ""]
        elif rd + cp > 0:
            if rd >= 2.0 * cp:
                verdict = ("acc_read carries it: the cost is the vendor HMX "
                           "drain. Dequant-from-VTCM\n  (1a) cannot shrink "
                           "this -- go to 1c (HVX GEMV for small M) and 2a "
                           "(T=512,\n  fewer P.V readouts).")
            elif cp >= 2.0 * rd:
                verdict = ("acc_copy carries it: the cost is on the DDR side, "
                           "which is our code.\n  1a (dequant straight from "
                           "the VTCM result tile) proceeds as planned.")
            else:
                verdict = ("acc_read and acc_copy split it -- both routes "
                           "needed: 1a for the copy\n  half, 1c/2a for the "
                           "vendor half.")
            out += ["  " + ink(verdict, 3), ""]
    return out


def write_html(rec, path, plain_sections):
    labels, values = [], []
    for regime, kv in shapes(rec):
        p = f"forward_{regime}_kv{kv}_I8I8"
        vs = [get(rec, "ATTN_STAGE", p, s) or 0.0 for s in STAGES]
        if sum(vs):
            labels.append(f"kv={kv} {'decode' if regime == 'dec' else 'prefill'}")
            values.append(vs)
    legend = " ".join(
        f'<span style="color:{STAGE_HTML[s]}">&#9632;</span> '
        f'{html.escape(STAGE_LABEL[s])}' for s in STAGES)
    body = f"""<title>HTP attention device report</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
        margin: 2rem auto; max-width: 940px; padding: 0 1rem; }}
h1 {{ font-size: 1.25rem; }}
pre {{ overflow-x: auto; padding: .75rem; border-radius: 6px;
       background: rgba(127,127,127,.12); }}
</style>
<h1>HTP attention &mdash; device run</h1>
<p>{legend}</p>
{bar_svg(labels, values)}
<pre>{html.escape(plain_sections)}</pre>
"""
    with open(path, "w") as fh:
        fh.write(body)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--html", metavar="FILE")
    ap.add_argument("--width", default="I8I8", choices=WIDTHS)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    use_color = (sys.stdout.isatty() and not args.no_color
                 and os.environ.get("NO_COLOR") is None)
    ink = Ink(use_color)
    cols = min(shutil.get_terminal_size((100, 24)).columns, 110)

    rec = parse(args.logs or DEFAULT_LOGS)
    if not rec:
        print("no marker lines found -- did the device run produce logs?",
              file=sys.stderr)
        return 1

    out = []
    out += header(ink, cols)
    out += env(rec, ink, cols)
    out += gates(rec, ink, cols)
    out += [""] + per_layer(rec, ink, cols)
    out += [""] + breakdown(rec, ink, cols, args.width)
    out += readout_split(rec, ink, cols, args.width)
    out += per_block(rec, ink, cols, args.width)
    out += [""]
    print("\n".join(out))

    if args.html:
        plain = Ink(False)
        sections = "\n".join(
            gates(rec, plain, 100) + [""] + per_layer(rec, plain, 100) + [""]
            + breakdown(rec, plain, 100, args.width)
            + readout_split(rec, plain, 100, args.width)
            + per_block(rec, plain, 100, args.width))
        write_html(rec, args.html, sections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
