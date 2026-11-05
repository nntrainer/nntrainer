#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the real DSP scratch cleanup/reserve functions with the real session type.

The allocator is the host libc; no SDK, device or copied implementation is used.
Hardware arithmetic and DSP/queue scheduling still require device validation.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    dsp = backend / "dsp"
    flags = ["-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-pthread"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
    with tempfile.TemporaryDirectory(prefix="nntr-prefill-scratch-") as path:
        temporary = Path(path)
        subprocess.run([sys.executable,
                        str(here / "gen_nntr_hvx_h.py"), str(dsp / "nntr_hvx.idl"),
                        str(temporary / "nntr_hvx.h")], check=True)
        binary = temporary / "prefill_scratch_host_check"
        command = shlex.split(os.environ.get("CC", "cc")) + flags
        command += ["-I" + str(p) for p in (temporary, here / "stub", dsp,
                                           backend, backend / "hmx", backend / "hvx")]
        command += [str(here / "prefill_scratch_host_check.c"),
                    str(dsp / "nntr_hvx_prefill_scratch.c"),
                    str(backend / "hmx/hexkl_moe_scratch.c"), "-o", str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == "__main__":
    main()
