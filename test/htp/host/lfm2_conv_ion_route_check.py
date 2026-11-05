#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the candidate's actual projection-dispatch block without a DSP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cxx", default="c++")
args = p.parse_args()
source = args.source.read_text()
begin = "  const bool row_resident = rows == 1 && htpDecodeRowResident(from);\n"
end = "  auto gate_pre = [&](size_t r) {\n"
assert source.count(begin) == source.count(end) == 1
block = source[source.index(begin):source.index(end)]
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "route.inc").write_text(block)
shutil.copy2(Path(__file__).with_name("lfm2_conv_ion_route_check.cpp"), args.output)
binary = args.output.resolve() / "route_check"
command = [args.cxx, "-std=c++17", "-O2", "-g", "-Wall", "-Wextra",
           "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
           str(args.output.resolve() / "lfm2_conv_ion_route_check.cpp"), "-o", str(binary)]
manifest = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "extracted_sha256": hashlib.sha256(block.encode()).hexdigest(),
            "compile": command, "run": [str(binary)]}
(args.output / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n")
subprocess.run(command, check=True)
subprocess.run([str(binary)], check=True)
