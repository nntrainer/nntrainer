#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the real DSP scratch cleanup/reserve functions with the real session type.

The allocator is the host libc; no SDK, device or copied implementation is used.
Hardware arithmetic and DSP/queue scheduling still require device validation.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"(?m)^(?:int|void) " + re.escape(name) + r"\(", source)
    if not match:
        raise ValueError("Missing production function: " + name)
    end = source.index("\n}\n", match.start()) + 3
    return source[match.start():end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    session = (here.parent / "hvx_add_f32.c").read_text()
    moe = (backend / "hmx/hexkl_mm_u8i4_moe.c").read_text()
    methods = "\n".join((function(moe, "hexkl_moe_scratch_free"),
                          function(moe, "hexkl_moe_scratch_reserve"),
                          function(session, "nntr_hvx_release_prefill_scratch")))
    flags = ["-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-pthread"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
    with tempfile.TemporaryDirectory(prefix="nntr-prefill-scratch-") as path:
        temporary = Path(path)
        (temporary / "prefill_scratch_methods.inc").write_text(methods)
        binary = temporary / "prefill_scratch_host_check"
        command = shlex.split(os.environ.get("CC", "cc")) + flags
        command += ["-I" + str(p) for p in (temporary, here / "stub", here.parent,
                                           backend, backend / "hmx", backend / "hvx")]
        command += [str(here / "prefill_scratch_host_check.c"), "-o", str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == "__main__":
    main()
