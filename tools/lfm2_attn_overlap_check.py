#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production Qx dequantizer and pipeline on SDK HVX emulation.

The original serial dequantizer is the exact reference. HMX/DMA/softmax/PV
are stand-ins; the production readout-address selection, worker dispatch,
block loop and dequant arithmetic are extracted without rewriting their bodies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    m = re.search(r"^static[^;{]*\b" + re.escape(name) + r"\([^;{]*\{", source, re.M)
    if not m:
        raise RuntimeError("missing production function " + name)
    end, depth = m.end(), 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[m.start():end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--source", type=Path, help="candidate hexkl_attn_q.c (defaults to --repo)")
    p.add_argument("--hexagon-tools", type=Path, default=os.environ.get("DEFAULT_HEXAGON_TOOLS_ROOT"))
    p.add_argument("--output", type=Path)
    p.add_argument("--sanitize", action="store_true")
    p.add_argument("--mutations", action="store_true")
    a = p.parse_args()
    if not a.hexagon_tools:
        p.error("set DEFAULT_HEXAGON_TOOLS_ROOT or pass --hexagon-tools")
    if a.output is None:
        a.output = Path(tempfile.mkdtemp(prefix="lfm2-attn-overlap-"))
    a.output.mkdir(parents=True, exist_ok=True)
    b = a.repo / "nntrainer/tensor/htp_backend"
    host = a.repo / "test/htp/host"
    if a.source is None:
        a.source = b / "hmx/hexkl_attn_q.c"
    fixture = Path(__file__).resolve().parents[1] / "test/htp/host/attn_overlap_native_check.c"
    source = a.source.read_text()
    (a.output / "source_manifest.json").write_text(json.dumps({
        "source": str(a.source.resolve()),
        "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "reference_source_commit": "0fcdfc1f9",
        "reference_source_sha256": "67a1cc02396fd4115d5d243b3563fa1260402e06bf96739e3dbe3260c2de6d52",
    }, indent=2) + "\n")
    start = source.index("typedef struct {\n  uint8_t *vb;")
    end = source.index("} attn_ctx;", start) + len("} attn_ctx;")
    jobend = source.index("} dequant_job;") + len("} dequant_job;")
    jobstart = source.rfind("typedef struct {", 0, jobend)
    h = """#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <AEEStdErr.h>
#include "hexkl_attn_q.h"
#include "hexkl_acc_tile.h"
#include "hvx_convert.h"
#define TB HEXKL_ATTN_TILE_BYTES
#define TILE_VECS (TB / sizeof(HVX_Vector))
""" + source[start:end] + "\n" + source[jobstart:jobend] + "\n"
    h += function((b / "hvx/hvx_tile_f16.h").read_text(), "hvx_tile_load_u") + "\n"
    for name in ("q_tile", "kt_tile", "s_tile", "acc_rows", "unit_range"):
        h += function(source, name) + "\n"
    for name in ("dequant_s_at", "dequant_s", "qx_dequant_available", "dequant_worker"):
        h += function(source, name) + "\n"
    h += """
static uint64_t now_us(void);
static void softmax_state_init(attn_ctx *, uint32_t, uint32_t);
static void dma_issue(attn_ctx *, uint32_t, uint32_t, uint32_t);
static void dma_land(attn_ctx *, uint32_t, uint32_t);
static void softmax_start(attn_ctx *, uint32_t, uint32_t, uint32_t, uint32_t, int);
static void softmax_finish_pquant(attn_ctx *, uint32_t);
static int phase_pv(attn_ctx *, uint32_t, int);
static void phase_store(attn_ctx *, uint32_t, uint32_t);
static void hexkl_micro_hmx_acc_clear_int32(void);
static void hexkl_hmx_mm_u8i8(const uint8_t *, const uint8_t *);
static void hexkl_hmx_mm_u8i4(const uint8_t *, const uint8_t *);
static int hexkl_micro_hmx_acc_read_int32(uint8_t *, uint32_t, uint32_t);
static void checked_submit(hvx_worker_pool *, hvx_worker_pool_func, void *, uint32_t);
static void checked_wait(hvx_worker_pool *);
"""
    h += function(source, "phase_qk") + "\n"
    h += "#define hvx_worker_pool_submit checked_submit\n#define hvx_worker_pool_wait checked_wait\n"
    h += function(source, "run_q_block") + "\n"
    h += "#undef hvx_worker_pool_submit\n#undef hvx_worker_pool_wait\n"
    generated = a.output / "attn_overlap_under_test.h"
    generated.write_text(h)
    native = a.hexagon_tools / "Tools/libnative"
    if not (native / "lib/libnative.a").is_file():
        p.error("missing Hexagon SDK host HVX emulation: " + str(native))
    flags = ["-std=gnu11", "-O2", "-g", "-fno-strict-aliasing", "-ffp-contract=off",
             "-include", "malloc.h", "-pthread", "-Wall", "-Wextra", "-Werror"]
    if a.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    includes = ["-isystem", str(native / "include")]
    for x in (host / "stub", b / "hvx", b / "hmx", a.output):
        includes += ["-I", str(x)]
    commands = []

    def run(cmd):
        commands.append(cmd)
        (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        subprocess.run(cmd, check=True)

    pool = a.output / "pool.o"
    run([os.environ.get("CC", "gcc"), *[x for x in flags if x != "-Werror"], *includes, "-c",
         str(b / "hvx/hvx_worker_pool.c"), "-o", str(pool)])

    def build_run(name, success=True):
        obj, exe = a.output / (name + ".o"), a.output / name
        run([os.environ.get("CC", "gcc"), *flags, *includes, "-c",
             str(fixture), "-o", str(obj)])
        run([os.environ.get("CXX", "g++"), *flags, str(obj), str(pool),
             str(native / "lib/libnative.a"), "-lm", "-o", str(exe)])
        result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=180)
        (a.output / (name + ".log")).write_text(result.stdout + result.stderr)
        if success:
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(result.stdout, end="", flush=True)
        elif result.returncode != -6 or "Assertion" not in result.stderr:
            raise RuntimeError("Mutation was not caught by an assertion: " + name)
        else:
            print("Mutation " + name + ": rejected as expected", flush=True)

    build_run("overlap_check")
    if a.mutations:
        # Mutate only the new deferred read address, leaving the serial path.
        bad = h.replace("c->L.qx_f32 + u * HEXKL_ATTN_Q_ACC_BYTES", "c->L.acc", 1)
        if bad == h:
            raise RuntimeError("read-address mutation target changed")
        generated.write_text(bad)
        build_run("wrong_accumulator", False)
        generated.write_text(h.replace("c->L.n_qb_chunk * c->t->g_br", "99u * c->t->g_br", 1)
                             .replace("raw_bytes <= (uint64_t)c->L.total - c->L.qx_f32", "raw_bytes > 0u", 1))
        build_run("missing_extent_guard", False)
        at = h.rfind("hvx_worker_pool_wait(c->pool);")
        if at < 0:
            raise RuntimeError("join mutation target changed")
        generated.write_text(h[:at] + "(void)c;" + h[at + len("hvx_worker_pool_wait(c->pool);"):])
        build_run("missing_pv_drain", False)
        generated.write_text(h)
    print("Artifacts: " + str(a.output), flush=True)


if __name__ == "__main__":
    main()
