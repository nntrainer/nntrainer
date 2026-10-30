#!/usr/bin/env python3
"""Verify signed Hadamard against the pre-change transform followed by XOR.

The portable build covers all fallback dimensions. SDK libnative runs the actual
1536/2048 HVX implementation; generated H44 inline assembly is replaced by abort
stubs in temporary copies, and no tested H12/Sylvester code is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HVX = Path("nntrainer/tensor/htp_backend/hvx")
EXPORTS = ("hvx_hadamard_supported", "hvx_hadamard_f32", "hvx_hadamard_rows_f32")
H44_STUBS = """#include <stdlib.h>
static void had44_32(const float *x, HVX_Vector *y, uint32_t s) { abort(); }
static void had44_pair(const float *x, HVX_Vector *y, uint32_t s) { abort(); }
static void had44_single(const float *x, HVX_Vector *y, uint32_t s) { abort(); }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hexagon-tools", type=Path, required=True)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    repo, reference = args.repo.resolve(), args.reference.resolve()
    source, out = (args.source or repo).resolve(), args.output.resolve()
    native = args.hexagon_tools.resolve() / "Tools/libnative"
    if not (native / "lib/libnative.a").is_file():
        parser.error("--hexagon-tools does not contain Tools/libnative/lib/libnative.a")
    out.mkdir(parents=True, exist_ok=True)
    (out / "native-types.h").write_text(
        "#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n"
        "typedef HVX_Vector nntr_unaligned_hvx __attribute__((aligned(1)));\n"
        "#define HVX_UVector nntr_unaligned_hvx\n#define __hexagon__ 1\n")
    flags = ["-std=gnu11", "-O1" if args.sanitize else "-O3", "-g",
             "-ffp-contract=off", "-fno-strict-aliasing",
             "-ffunction-sections", "-fdata-sections"]
    link_flags = ["-Wl,--gc-sections"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                  "-fno-omit-frame-pointer", "-fno-pie"]
        link_flags += ["-fsanitize=address,undefined", "-no-pie"]
    includes = ["-I", str(native / "include"), "-I", str(source / HVX),
                "-I", str(repo / HVX)]
    cc, cxx = os.environ.get("CC", "gcc"), os.environ.get("CXX", "g++")
    commands = []

    def run(command, label):
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        log_path = out / (label + ".log")
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{label} failed ({result.returncode}); see {log_path}")
        return log_path.read_text()

    for use_native in (False, True):
        suffix = "native" if use_native else "portable"
        objects = []
        for name, root in (("reference", reference), ("candidate", source)):
            text = (root / HVX / "hvx_hadamard_f32.c").read_text()
            if use_native:
                include = '#include "hvx_hadamard_h44.inc"'
                if text.count(include) != 1:
                    raise SystemExit("H44 assembly include anchor changed")
                text = text.replace(include, H44_STUBS)
            src = out / (name + "-" + suffix + ".c")
            src.write_text(text)
            obj = src.with_suffix(".o")
            objects.append(str(obj))
            definitions = (["-D" + s + "=reference_" + s for s in EXPORTS]
                           if name == "reference" else [])
            native_flags = ["-include", str(out / "native-types.h")] if use_native else []
            run([cc, *flags, *includes, *native_flags, *definitions, "-c", str(src),
                 "-o", str(obj)], "build-" + name + "-" + suffix)
        fixture = Path(__file__).resolve().with_name("hadamard_signed_check.c")
        obj = out / ("fixture-" + suffix + ".o")
        run([cc, *flags, *includes, "-c", str(fixture), "-o", str(obj)],
            "fixture-" + suffix)
        exe = out / ("check-" + suffix)
        library = [str(native / "lib/libnative.a")] if use_native else []
        run([cxx, *link_flags, str(obj), *objects, *library, "-lm", "-o", str(exe)],
            "link-" + suffix)
        print(run([str(exe)], "run-" + suffix), end="")
    hashes = {name: hashlib.sha256((root / HVX / "hvx_hadamard_f32.c").read_bytes()).hexdigest()
              for name, root in (("reference", reference), ("candidate", source))}
    (out / "results.json").write_text(json.dumps({
        "pass": True, "sanitize": args.sanitize, "source_sha256": hashes,
        "limitations": ["Generated H44 assembler uses unreachable abort stubs in libnative copies.",
                        "Native arithmetic/canaries are verified; no DSP timing or cache simulation."]},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
