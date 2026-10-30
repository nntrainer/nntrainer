#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run production Q-prep and HVX quantizers on the SDK's host emulation.

HMX is outside this test. The unmodified candidate Q-prep body is extracted so
the real guard, initializers, gather and helper calls are exercised. Hexagon
libnative supplies HVX arithmetic; the repository's QuRT host shim supplies
worker threads. No model file, device, benchmark or absolute path is required.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^static[^;{]*\b" + re.escape(name) + r"\([^;{]*\{", source, re.M)
    if not match:
        raise RuntimeError(f"Cannot locate production function {name}")
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hexagon-tools", type=Path,
                        default=os.environ.get("DEFAULT_HEXAGON_TOOLS_ROOT"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--mutations", action="store_true")
    args = parser.parse_args()
    if not args.hexagon_tools:
        parser.error("set DEFAULT_HEXAGON_TOOLS_ROOT or --hexagon-tools")
    native = args.hexagon_tools / "Tools/libnative"
    if not (native / "lib/libnative.a").is_file():
        parser.error(f"missing SDK HVX emulation: {native}")
    out = args.output_dir or Path(tempfile.mkdtemp(prefix="lfm2-qprep-check-"))
    out.mkdir(parents=True, exist_ok=True)
    backend = args.repo / "nntrainer/tensor/htp_backend"
    host = args.repo / "test/htp/host"
    source = (backend / "hmx/hexkl_attn_q.c").read_text()
    begin = source.index("typedef struct {\n  uint8_t *vb;")
    end = source.index("} attn_ctx;", begin) + len("} attn_ctx;")
    tile = (backend / "hvx/hvx_tile_f16.h").read_text()
    header = """#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <AEEStdErr.h>
#include "hexkl_attn_q.h"
#define TB HEXKL_ATTN_TILE_BYTES
typedef struct unused_acc_layout hexkl_acc_layout;
""" + function(tile, "hvx_tile_load_u") + "\n" + source[begin:end] + "\n"
    for name in ("q_tile", "qx_row", "unit_range"):
        header += function(source, name) + "\n"
    gather_begin = source.index("typedef struct {\n  attn_ctx *c;")
    gather_end = source.index("} qprep_ctx;", gather_begin) + len("} qprep_ctx;")
    header += source[gather_begin:gather_end] + "\n"
    header += function(source, "qprep_gather_worker") + "\n"
    header += """#define hvx_quant_rows_u8_params checked_params
#define hvx_quant_pack_u8_ah checked_pack
#define hvx_quant_params_pack_u8_ah checked_fused
""" + function(source, "phase_qprep") + """
#undef hvx_quant_rows_u8_params
#undef hvx_quant_pack_u8_ah
#undef hvx_quant_params_pack_u8_ah
"""
    generated = out / "attn_qprep_under_test.h"
    generated.write_text(header)
    cc = os.environ.get("CC", "gcc")
    cxx = os.environ.get("CXX", "g++")
    flags = ["-std=gnu11", "-O1", "-g", "-fno-strict-aliasing", "-ffp-contract=off",
             "-include", str(host / "attn_qprep_native_unaligned.h"),
             "-include", "malloc.h", "-pthread"]
    san = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
           "-fno-omit-frame-pointer"] if args.sanitize else []
    includes = ["-isystem", str(native / "include")]
    includes += ["-I" + str(p) for p in (host / "stub", backend / "hvx",
                                        backend / "hmx", out)]
    objects = []
    for name in ("hvx_quant_u8", "hvx_rmsnorm_rows_f32", "hvx_worker_pool"):
        obj = out / (name + ".o")
        subprocess.run([cc, *flags, *san, *includes, "-c", str(backend / "hvx" / (name + ".c")),
                        "-o", str(obj)], check=True)
        objects.append(str(obj))

    def build_run(name, expected_success=True):
        obj = out / (name + ".o")
        exe = out / name
        subprocess.run([cc, *flags, *san, "-Wall", "-Wextra", "-Werror", *includes,
                        "-c", str(host / "attn_qprep_native_check.c"), "-o", str(obj)], check=True)
        subprocess.run([cxx, "-pthread", *san, str(obj), *objects,
                        str(native / "lib/libnative.a"), "-lm", "-o", str(exe)], check=True)
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        env["UBSAN_OPTIONS"] = "halt_on_error=1"
        proc = subprocess.run([str(exe)], text=True, capture_output=True,
                              timeout=180, env=env)
        (out / (name + ".log")).write_text(proc.stdout + proc.stderr)
        if (proc.returncode == 0) != expected_success:
            raise RuntimeError(f"Unexpected status {proc.returncode}: {out / (name + '.log')}")
        if not expected_success and (proc.returncode != -6 or "Assertion" not in proc.stderr):
            raise RuntimeError(f"Mutation failed outside an assertion: {name}")
        print(proc.stdout.strip() if expected_success else f"Mutation {name}: rejected as expected", flush=True)

    build_run("attn_qprep_native_check")
    if args.mutations:
        for name, before, after in (
            ("missing_zero_init", "c->q_scale[r] = 1.0f;", "c->q_scale[r] = -37.0f;"),
            ("wrong_kind_guard", "c->kv->kind == HEXKL_KV_Q8", "c->kv->kind <= HEXKL_KV_Q4"),
        ):
            if header.count(before) != 1:
                raise RuntimeError(f"Mutation target changed: {before}")
            generated.write_text(header.replace(before, after))
            build_run(name, False)
        generated.write_text(header)
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
