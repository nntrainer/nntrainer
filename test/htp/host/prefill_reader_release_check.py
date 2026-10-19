#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production prefill reader lifecycle with deterministic workers."""
import argparse
from pathlib import Path
import shutil
import subprocess


def method(source, name):
    begin = source.index("  void " + name + "()")
    pos = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[begin:pos] + "\n"


p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cxx", default="c++")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
source = (a.repo / "nntrainer/tensor/htp_backend/htp_compute_ops.cpp").read_text()
for name in ("releaseIdlePrefillReaders", "startPrefetchReaders"):
    (a.output / (name + ".inc")).write_text(method(source, name))
fixture = Path(__file__).with_suffix(".cpp")
shutil.copy2(fixture, a.output / fixture.name)
binary = a.output / "prefill_reader_release_check"
subprocess.run([a.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-pthread", str(a.output / fixture.name), "-o", str(binary)],
               check=True)
subprocess.run([str(binary)], check=True, timeout=30)
