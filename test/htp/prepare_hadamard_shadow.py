#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a separate, diagnostic-only decode rotation build.

This is NOT a rotated model: prefill stays unrotated; decode runs the real
Hadamard kernels, then restores each input before quantization. Thus the app
keeps the original checkpoint's activations, routing and file reads. The
measured delta includes the extra preservation copies. Use the real binary
for actual rotation timings and numerical validation; never pool the two.

The repository kernel is not modified. The generated build script writes
build/libnntr_hvx_hadamard_shadow.so, leaving the normal skel intact. It uses
the same SDK environment as build.sh. Only Gemma's M=1, 2816/704 is supported.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"source changed: expected one occurrence of {before!r}")
    return text.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    source = root / "nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c"
    original = source.read_text()
    text = replace_once(original, "static void moe_rotate_input_worker(", """
/* DIAGNOSTIC ONLY: actual full-width FP32 HVX rotation followed by restore.
 * noinline keeps the preservation buffer's lifetime separate from the
 * calling worker. Neither the call nor its side effects may be elided. */
static __attribute__((noinline)) void shadow_rotate_restore(float *row,
                                                            uint32_t width,
                                                            uint32_t mode) {
  float saved[2816] __attribute__((aligned(128)));
  memcpy(saved, row, (size_t)width * sizeof(float));
  (void)hvx_hadamard_f32(row, width, mode);
  memcpy(row, saved, (size_t)width * sizeof(float));
}

static void moe_rotate_input_worker(""")
    # Stage B precedes the helper's definition.
    text = replace_once(text, "static void moe_m1_requant_worker(", """
static void shadow_rotate_restore(float *row, uint32_t width, uint32_t mode);
static void moe_m1_requant_worker(""")
    text = replace_once(text,
        "    moe_rotate_rows(e->gate_f32, e->m, c->inter, c->had_down);",
        "    if (c->had_down)\n"
        "      shadow_rotate_restore(e->gate_f32, c->inter, c->had_down);")
    text = replace_once(text,
        "      (void)hvx_hadamard_f32(row, c->K, c->mode);",
        "      shadow_rotate_restore(row, c->K, c->mode);")
    text = replace_once(text, "  const uint32_t act =\n", """
  /* Decode-only control: the same prefill/KV seed as the off variant. */
  if (M != 1u)
    flags &= ~(HEXKL_MOE_FLAG_HAD_UP | HEXKL_MOE_FLAG_HAD_DOWN |
               HEXKL_MOE_FLAG_HAD_SCALAR | HEXKL_MOE_FLAG_HAD_HMX);
  const uint32_t act =
""")
    text = replace_once(text, "  int use_had_hmx = 0;", """
  if ((flags & (HEXKL_MOE_FLAG_HAD_UP | HEXKL_MOE_FLAG_HAD_DOWN)) &&
      (M != 1u || K != 2816u || inter != 704u))
    return AEE_EUNSUPPORTED;
  int use_had_hmx = 0;""")
    # Reject a repeated-row or HMX fallback call: the shadow Stage B above
    # intentionally supports exactly one activation row per routed expert.
    text = replace_once(text,
        "  const int use_m1 = m1_shape && (allow_tail || all_2bit);",
        "  const int use_m1 = m1_shape && (allow_tail || all_2bit);\n"
        "  if ((had_up || had_down) && (!use_m1 || widest != 1u))\n"
        "    return AEE_EUNSUPPORTED;")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    generated = out / "hexkl_mm_u8i4_moe_shadow.c"
    generated.write_text(text)
    (out / "shadow.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True),
        fromfile=str(source.relative_to(root)), tofile=generated.name)))
    build = (here / "build.sh").read_text()
    build = replace_once(build,
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "SCRIPT_DIR=" + shlex.quote(str(here)))
    # SRCS is intentionally word-split by the existing builder. Require a
    # path without shell metacharacters rather than invent nested quoting.
    if shlex.quote(str(generated)) != str(generated):
        raise ValueError("use an output directory without shell metacharacters")
    build = replace_once(build, "$BACKEND/hmx/hexkl_mm_u8i4_moe.c",
                         str(generated))
    build = build.replace("build/libnntr_hvx_skel.so",
                          "build/libnntr_hvx_hadamard_shadow.so")
    (out / "build.sh").write_text(build)
    (out / "provenance.json").write_text(json.dumps({
        "purpose": "diagnostic decode rotation + preserve/restore copies",
        "normal_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "shadow_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prefill": "rotation disabled", "decode": "rotate then restore",
        "output_binary": str(here / "build/libnntr_hvx_hadamard_shadow.so"),
        "not_for_inference": True,
    }, indent=2) + "\n")
    print("Diagnostic build: bash " + shlex.quote(str(out / "build.sh")))


if __name__ == "__main__":
    main()
