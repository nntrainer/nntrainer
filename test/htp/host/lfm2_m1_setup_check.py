#!/usr/bin/env python3
"""Build actual before/after whole-MoE TUs with observation-only setup hooks.

Requires the repository's existing host stand-ins and a GNU-compatible C
compiler. --reference is the original hexkl_mm_u8i4_moe.c, supplied explicitly.
No production source is rewritten. Generated sources/objects live in --output.
The actual production quantizer runs through the SDK's libnative HVX emulator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


EXPORTS = (
    "hexkl_moe_scratch_free", "hexkl_moe_scratch_reserve", "hexkl_moe_carve",
    "hexkl_moe_push_weight_chunk", "hexkl_moe_expand_chunk",
    "hexkl_moe_push_act_block", "hexkl_moe_pack_bg_worker",
    "hexkl_moe_dma_copy", "hexkl_mm_u8i4_moe_layout",
    "hexkl_mm_u8i4_moe_layer_run", "hexkl_mm_u8i4_fc_m1_run",
)
DECL = """#include <stdint.h>
#include <stddef.h>
void lfm2_m1_setup_observe(uint32_t, uint32_t, uint32_t, uint32_t,
 const uint32_t *, const uint32_t *, const float *, const int32_t *,
 const float *, const int32_t *, const uint8_t *, size_t, const float *, int);
"""
ANCHOR = "    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);\n    if (gu0_async) {"
HOOK = """    lfm2_m1_setup_observe(n_active, n_slots, K, N_out, slot_of,
      slot_row, slot_scale, slot_zp, scale_all, zp_all, pack_skip,
      sz_pack_done, out_c, rows_out);
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--source", type=Path, help="candidate source override for mutation checks")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--hexagon-tools", required=True, type=Path)
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--mutations", action="store_true",
                    help="require three setup-consumer/zeroing mutants to fail")
    args = ap.parse_args()
    if args.mutations and not args.sanitize:
        ap.error("--mutations requires --sanitize")
    root, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = root / "nntrainer/tensor/htp_backend"
    here = root / "test/htp/host"
    candidate = args.source or backend / "hmx/hexkl_mm_u8i4_moe.c"
    native = args.hexagon_tools.resolve() / "Tools/libnative"
    if not (native / "lib/libnative.a").is_file():
        raise SystemExit("--hexagon-tools must contain Tools/libnative/lib/libnative.a")
    (out / "observe.h").write_text(DECL)
    sources = {}
    for key, source in (("reference", args.reference), ("candidate", candidate)):
        text = source.read_text()
        if text.count(ANCHOR) != 1:
            raise SystemExit(f"{source}: setup anchor changed")
        sources[key] = hashlib.sha256(source.read_bytes()).hexdigest()
        (out / f"{key}.c").write_text(text.replace(ANCHOR, HOOK + ANCHOR))
    cc = os.environ.get("CC", "gcc")
    flags = ["-std=gnu11", "-O1" if args.sanitize else "-O2", "-g",
             "-Wall", "-Wextra", "-Wno-unused-parameter", "-ffp-contract=off"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                  "-fno-omit-frame-pointer", "-fno-pie"]
    for directory in (here / "stub", here / "standin", here, here.parent,
                      backend.parent, backend, backend / "hmx", backend / "hvx"):
        flags += ["-I", str(directory)]
    commands = []

    def run(command, name, failure_marker=None):
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if failure_marker:
            if not result.returncode or failure_marker not in (out / f"{name}.log").read_text():
                raise SystemExit(f"{name}: mutant did not fail for {failure_marker!r}")
        elif result.returncode:
            raise SystemExit(f"{name} failed ({result.returncode}); see {out / (name + '.log')}")

    objects = []
    for key in ("reference", "candidate"):
        obj = out / f"{key}.o"
        definitions = ([f"-D{symbol}=reference_{symbol}" for symbol in EXPORTS]
                       if key == "reference" else [])
        run([cc, *flags, *definitions, "-include", str(out / "observe.h"),
             "-c", str(out / f"{key}.c"), "-o", str(obj)], f"build-{key}")
        objects.append(obj)
    for source in (here / "lfm2_m1_setup_check.c", here / "standin/hvx_scalar.c",
                   backend / "hmx/hexkl_dma_trace.c", backend / "hvx/hvx_expand_i2i4.c",
                   backend / "hvx/hvx_gemv_u8tl2.c"):
        obj = out / (source.stem + ".o")
        definitions = []
        if source.name == "hvx_scalar.c":
            definitions = [f"-D{symbol}=unused_standin_{symbol}" for symbol in (
                "hvx_quant_rows_u8_params", "hvx_quant_rows_u8_params_rm",
                "hvx_quant_pack_u8_ah_rows", "hvx_quant_pack_u8_ah_m1_repeat",
                "hvx_quant_pack_u8_ah_mapped", "hvx_quant_pack_u8_ah",
                "hvx_quant_params_pack_u8_ah",
                "hvx_gather_u8_ah_rows",
            )]
            if args.sanitize:
                # Existing scalar swiglu_det_exp constructs an exponent with
                # a signed left shift (swiglu_det.h:152). Exclude only that
                # sanitizer class in the stand-in, retaining all sanitizers
                # for both production MoE TUs and the actual quantizer.
                definitions += ["-fno-sanitize=shift-base"]
        run([cc, *flags, *definitions, "-c", str(source), "-o", str(obj)],
            "build-" + source.stem)
        objects.append(obj)
    obj = out / "hvx_quant_u8.o"
    run([cc, *flags, "-fno-strict-aliasing", "-DHVX_UVector=HEXAGON_Vect1024",
         "-include", "malloc.h", "-I", str(native / "include"), "-c",
         str(backend / "hvx/hvx_quant_u8.c"), "-o", str(obj)], "build-hvx_quant_u8")
    objects.append(obj)
    exe = out / "lfm2_m1_setup_check"
    run([os.environ.get("CXX", "g++"), *flags, *(["-no-pie"] if args.sanitize else []),
         *map(str, objects), str(native / "lib/libnative.a"), "-lm", "-o", str(exe)], "link")
    run([str(exe)], "check")
    (out / "source-hashes.json").write_text(json.dumps(sources, indent=2) + "\n")
    print((out / "check.log").read_text(), end="")
    if args.mutations:
        mutations = (
            ("read_padding", "x->act_scale = slot_scale + sb;",
             "x->act_scale = slot_scale + sb + 1u;", "use-after-poison"),
            ("missing_row_zero", "memset(y, 0, sizeof(float) * N_out);",
             "(void)y; /* removed required output zero */", "Assertion"),
            ("fallback_clear", "if (!lfm_m1_setup || !rows_out)",
             "if (!rows_out)", "Assertion"),
        )
        original = candidate.read_text()
        for name, old, new, marker in mutations:
            if original.count(old) != 1:
                raise SystemExit(f"{name}: source anchor changed")
            mutated = original.replace(old, new).replace(ANCHOR, HOOK + ANCHOR)
            source = out / f"mutant-{name}.c"
            source.write_text(mutated)
            obj = out / f"mutant-{name}.o"
            run([cc, *flags, "-include", str(out / "observe.h"), "-c",
                 str(source), "-o", str(obj)], "build-mutant-" + name)
            mutant_objects = [obj if p.name == "candidate.o" else p for p in objects]
            mutant = out / f"mutant-{name}"
            run([os.environ.get("CXX", "g++"), *flags, "-no-pie",
                 *map(str, mutant_objects), str(native / "lib/libnative.a"),
                 "-lm", "-o", str(mutant)], "link-mutant-" + name)
            run([str(mutant)], "mutant-" + name, marker)
            print(f"Mutation {name}: required {marker} detected")


if __name__ == "__main__":
    main()
