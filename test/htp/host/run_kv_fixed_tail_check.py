#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract and check actual fixed-Q8 DSP copy/worker under SDK libnative.

Linux host test, using the repository's pthread QuRT stand-in. No device IO.
The baseline overflow/order failure is an expected, separately logged result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def function(text, name):
    m = re.search(r"^static[^;{]*\b" + re.escape(name) + r"\([^;{]*\{", text, re.M)
    if not m:
        raise RuntimeError("missing production function " + name)
    i, depth = m.end(), 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[m.start():i]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--hexagon-tools", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not sys.platform.startswith("linux"):
        p.error("this optional libnative/pthread fixture requires a Linux host")
    a.output.mkdir(parents=True, exist_ok=True)
    b = a.repo / "nntrainer/tensor/htp_backend"
    host = a.repo / "test/htp/host"
    native = a.hexagon_tools / "Tools/libnative"
    fixture = Path(__file__).resolve().with_name("kv_fixed_tail_check.c")
    flags = ["-std=gnu11", "-O2", "-g", "-pthread", "-Wall", "-Wextra", "-Werror",
             "-fno-strict-aliasing", "-ffp-contract=off", "-include", "malloc.h",
             "-include", str(host / "attn_qprep_native_unaligned.h"),
             "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    inc = ["-isystem", str(native / "include"), "-I", str(host / "stub"),
           "-I", str(b / "hvx"), "-I", str(b / "hmx")]
    env = dict(os.environ, ASAN_OPTIONS="halt_on_error=1:detect_leaks=1",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    commands, results = [], {}

    def run(cmd, label, expected=None):
        commands.append(cmd)
        (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        q = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        output = q.stdout + q.stderr
        (a.output / (label + ".log")).write_text(output)
        results[label] = {"returncode": q.returncode,
                          "expected_failure": expected is not None}
        if expected is not None:
            if q.returncode == 0 or expected not in output:
                raise RuntimeError(label + " did not show expected defect:\n" + output)
        elif q.returncode:
            raise RuntimeError(label + " failed:\n" + output)
        return output

    objects = []
    for name, src in (("metadata", a.reference),
                      ("quant", b / "hvx/hvx_kv_quant.c"),
                      ("pool", b / "hvx/hvx_worker_pool.c")):
        obj = a.output / (name + ".o")
        # Existing pthread stand-in has an unused parameter; no suppression of
        # any sanitizer, and candidate fixture itself retains -Werror.
        f = [x for x in flags if x != "-Werror"] if name == "pool" else flags
        run([os.environ.get("CC", "gcc"), *f, *inc, "-c", str(src), "-o", str(obj)], name + "-build")
        objects.append(obj)
    for label, src in (("baseline", a.reference), ("candidate", a.source)):
        sub = a.output / label
        sub.mkdir(exist_ok=True)
        text = src.read_text()
        end = text.index("} quant_rows_ctx;") + len("} quant_rows_ctx;")
        start = text.rfind("typedef struct {", 0, end)
        h = '''#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "hexkl_kv_q.h"
#include "hvx_kv_quant.h"
#include "hvx_tile_f16.h"
'''
        h += function(text, "xor80_copy") + "\n" + text[start:end] + "\n"
        h += function(text, "quant_rows_worker") + "\n"
        (sub / "kv_fixed_tail_under_test.h").write_text(h)
        obj, exe = sub / "fixture.o", sub / "check"
        run([os.environ.get("CC", "gcc"), *flags, *inc, "-I", str(sub),
             "-c", str(fixture), "-o", str(obj)], label + "-build")
        run([os.environ.get("CXX", "g++"), *flags, str(obj), *map(str, objects),
             str(native / "lib/libnative.a"), "-lm", "-o", str(exe)], label + "-link")
        if label == "baseline":
            run([str(exe), "--copy-overflow"], "baseline-copy-overflow",
                "AddressSanitizer: heap-buffer-overflow")
            run([str(exe), "--worker-order"], "baseline-worker-order",
                "Assertion")
        else:
            print(run([str(exe)], "candidate-check"), end="", flush=True)
            run([str(exe), "--copy-overflow"], "candidate-copy64")
            run([str(exe), "--worker-order"], "candidate-worker-order")
    results["hashes"] = {str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                         for x in (a.source, a.reference, fixture, Path(__file__))}
    results["scope"] = "actual DSP helper and row worker, actual HVX/libnative quant, host registration, pthread pool; not HMX/device"
    results["sanitizers"] = "ASan + UBSan halt-on-error, no suppression"
    (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
