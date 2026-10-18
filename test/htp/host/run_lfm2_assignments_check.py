#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the actual token assignment method with a serial source revision."""
import argparse
from pathlib import Path
import subprocess
from run_lfm2_router_arm_native import function

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
p.add_argument("--baseline", default="8552240c419ae0797d7f382384880d58e905f123")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rel = "Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp"
old = subprocess.check_output(["git", "show", a.baseline + ":" + rel], cwd=a.repo, text=True)
new = (a.repo / rel).read_text()
start = "void Lfm2MoELayer::buildExpertAssignments("
for name, source in (("reference", old), ("candidate", new)):
    (a.output / (name + ".inc")).write_text(function(source, start).replace(
        "Lfm2MoELayer::buildExpertAssignments", "Layer::" + name))
fixture = Path(__file__).with_name("lfm2_assignments_check.cpp")
exe = a.output / "check"
cmd = ["g++", "-std=c++17", "-O2", "-g", "-pthread", "-ffp-contract=off",
       "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
       "-Wall", "-Wextra", "-Werror", "-I" + str(a.output), str(fixture), "-o", str(exe)]
subprocess.run(cmd, check=True)
subprocess.run([str(exe)], check=True)
