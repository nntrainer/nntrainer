#!/usr/bin/env python3
"""Check actual TL2 expansion and extracted caller units with libnative HVX.

--reference and --source are explicit repository/overlay roots. The reference
must contain the pre-change expansion C file and MoE C file. No input source is
modified. This checks bytes and selection, not device performance or DMA timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HVX = Path("nntrainer/tensor/htp_backend/hvx")
MOE = Path("nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c")
EXPORTS = ("hvx_tl2_weight_bytes", "hvx_tl2_table_bytes", "hvx_tl2_k_splits",
           "hvx_tl2_pack", "hvx_tl2_unpack", "hvx_tl2_expand_wh96",
           "hvx_tl2_prepare", "hvx_gemv_u8tl2_panel")


def unit(text, name):
    end = text.index("} moe_expand_ctx;") + len("} moe_expand_ctx;")
    start = text.rfind("typedef struct {", 0, end)
    body_start = text.index("static void moe_expand_unit(")
    body_end = text.index("\n/** Called only by the issuing thread", body_start)
    value = text[start:end] + "\n" + text[body_start:body_end]
    value = value.replace("moe_expand_ctx", name + "_expand_ctx")
    value = value.replace("moe_expand_unit", name + "_expand_unit")
    if name == "reference":
        value = value.replace("hvx_tl2_expand_wh96(",
                              "reference_hvx_tl2_expand_wh96(")
    else:
        value = value.replace("hvx_tl2_expand_wh96_disjoint(", "disjoint_expand(")
        value = value.replace("hvx_tl2_expand_wh96(", "original_expand(")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hexagon-tools", type=Path, required=True,
                        help="SDK tools/HEXAGON_Tools/<version> directory")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--mutations", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = (args.source or repo).resolve()
    reference, out = args.reference.resolve(), args.output.resolve()
    native = args.hexagon_tools.resolve() / "Tools/libnative"
    out.mkdir(parents=True, exist_ok=True)
    if not (native / "lib/libnative.a").is_file():
        parser.error("--hexagon-tools does not contain Tools/libnative/lib/libnative.a")
    if args.mutations and not args.sanitize:
        parser.error("--mutations requires --sanitize")
    source_text = (source / MOE).read_text()
    (out / "reference-unit.inc").write_text(unit((reference / MOE).read_text(), "reference"))
    (out / "candidate-unit.inc").write_text(unit(source_text, "candidate"))
    # libnative supplies HVX_Vector but not HVX_UVector. Match the production
    # unaligned type explicitly instead of disabling alignment sanitization.
    (out / "native-types.h").write_text(
        "#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n"
        "typedef HVX_Vector nntr_unaligned_hvx __attribute__((aligned(1)));\n"
        "#define HVX_UVector nntr_unaligned_hvx\n#define __hexagon__ 1\n")
    cc, cxx = os.environ.get("CC", "gcc"), os.environ.get("CXX", "g++")
    flags = ["-std=gnu11", "-O1" if args.sanitize else "-O3", "-g",
             "-Wall", "-Wextra", "-Wno-unused-parameter", "-fno-strict-aliasing",
             "-ffunction-sections", "-fdata-sections",
             "-include", "malloc.h", "-include", str(out / "native-types.h")]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                  "-fno-omit-frame-pointer", "-fno-pie"]
    for directory in (out, native / "include", source / HVX, repo / HVX,
                      repo / "test/htp/host/stub"):
        flags += ["-I", str(directory)]
    commands = []

    def run(command, label, expect=None):
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        path = out / (label + ".log")
        with path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        value = path.read_text()
        if expect:
            if result.returncode == 0 or expect not in value:
                raise SystemExit(f"{label}: expected rejection {expect!r}; see {path}")
        elif result.returncode:
            raise SystemExit(f"{label}: failed ({result.returncode}); see {path}")
        return value

    objects = []
    for name, root in (("reference", reference), ("candidate", source)):
        obj = out / (name + ".o")
        definitions = (["-D" + s + "=reference_" + s for s in EXPORTS]
                       if name == "reference" else [])
        run([cc, *flags, *definitions, "-c", str(root / HVX / "hvx_gemv_u8tl2.c"),
             "-o", str(obj)], "build-" + name)
        objects.append(obj)
    expand_obj = out / "expand-i2i4.o"
    run([cc, *flags, "-c", str(repo / HVX / "hvx_expand_i2i4.c"),
         "-o", str(expand_obj)], "build-expand-i2i4")
    objects.append(expand_obj)
    fixture = Path(__file__).resolve().with_name("tl2_expand_disjoint_check.c")
    check_obj = out / "check.o"

    def build_fixture(suffix):
        run([cc, *flags, "-c", str(fixture), "-o", str(check_obj)],
            "build-check" + suffix)
        exe = out / ("check" + suffix)
        link_flags = ["-Wl,--gc-sections"]
        if args.sanitize:
            link_flags += ["-fsanitize=address,undefined", "-no-pie"]
        run([cxx, *link_flags, str(check_obj), *map(str, objects),
             str(native / "lib/libnative.a"), "-lm", "-o", str(exe)],
            "link" + suffix)
        return exe

    print(run([str(build_fixture(""))], "run"), end="")
    mutants = []
    if args.mutations:
        for name, old, new in (
            ("snapshot-selection", "c->native_packed &&", "1 &&"),
            ("gu-stride", "c->stride == 96u * 512u", "1"),
            ("dn-stride", "c->stride == 64u * 512u", "1"),
        ):
            if source_text.count(old) != 1:
                raise SystemExit(f"{name}: production anchor changed")
            changed = source_text.replace(old, new)
            (out / "candidate-unit.inc").write_text(unit(changed, "candidate"))
            run([str(build_fixture("-" + name))], "mutant-" + name,
                "FAIL: caller selection")
            mutants.append(name)
            print("PASS: mutant rejected:", name)
        (out / "candidate-unit.inc").write_text(unit(source_text, "candidate"))
    hashes = {}
    for name, root in (("reference", reference), ("candidate", source)):
        hashes[name] = {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in (MOE, HVX / "hvx_gemv_u8tl2.c", HVX / "hvx_gemv_u8tl2.h")}
    (out / "results.json").write_text(json.dumps({
        "pass": True, "mutants_rejected": mutants, "sources": hashes,
        "limitations": ["HVX arithmetic uses SDK libnative; DMA/timing are not simulated.",
                        "Eight actual caller units execute in permuted serial order; no worker-pool race claim.",
                        "Only complete caller type and expansion-unit body are extracted; source-level dispatch unchanged."]},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
