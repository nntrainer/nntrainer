"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

@file check_arm_isa_config.py
@date 27 November 2026
@brief Configure nntrainer with ARM ISA option combinations and compare the
       generated flags and sources with the expected values
@author Jaemin Shin <jaemin980311@gmail.com>

Runs `meson setup` only (no compile). For each case it checks:
  - configure succeeds, or fails with the expected error
  - the ISA warnings printed are exactly the expected ones
  - every nntrainer compile command has the expected -march (and -mfpu)
  - no -D__ARM_FEATURE_* is forced on the command line
  - KleidiAI dotprod / i8mm kernel sources are built only when enabled

Suites:
  aarch64        Linux aarch64 cross build; compiler must know sve2
  aarch64-nosme  Linux aarch64 cross build with a compiler without +sme
  armhf          Linux 32-bit arm cross build
  x86            native x86_64 build
  android        Android build; ndk-build must be on PATH

Example:
  python3 tools/pyutils/check_arm_isa_config.py --suite aarch64 \\
      --cross-prefix aarch64-linux-gnu-
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Options that keep configure fast and free of target libraries
COMMON_OPTS = [
    "-Denable-test=false",
    "-Denable-app=false",
    "-Denable-tflite-interpreter=false",
    "-Denable-tflite-backbone=false",
]
# The Linux builds would cmake OpenBLAS at configure time; Android needs blas
LINUX_OPTS = ["-Denable-blas=false"]
ANDROID_OPTS = ["-Dplatform=android"]

# Every ISA warning the root meson.build and jni/meson.build can print
ISA_WARNINGS = [
    "enable-fp16 without enable-dotprod",
    "enable-dotprod without enable-i8mm",
    "enable-i8mm without enable-dotprod",
    "is ignored on non-aarch64 builds",
    "is ignored on non-ARM builds",
    "NDK clang not found",
    "FP16 is enabled on armv7",
    "is ignored on this build",
]

V82 = "-march=armv8.2-a"
DEF = V82 + "+dotprod+i8mm"


def case(name, opts, march=None, kai=None, warn=(), error=None, ndk=None):
    """kai: None (not checked), 'none', 'dotprod' or 'i8mm' (dotprod+i8mm).
    ndk: None (real NDK), 'reject' (clang fails) or 'noclang'."""
    return {
        "name": name,
        "opts": opts,
        "march": march,
        "kai": kai,
        "warn": list(warn),
        "error": error,
        "ndk": ndk,
    }


AARCH64 = [
    case("default", [], DEF, "i8mm"),
    case("armv8-a", ["-Darm-arch=armv8-a"], "-march=armv8-a+dotprod+i8mm",
         "i8mm"),
    case("armv8-a plain",
         ["-Darm-arch=armv8-a", "-Denable-dotprod=false",
          "-Denable-i8mm=false"], "-march=armv8-a", "none"),
    case("no dotprod", ["-Denable-dotprod=false"], V82 + "+i8mm", "none",
         warn=["enable-i8mm without enable-dotprod"]),
    case("no i8mm", ["-Denable-i8mm=false"], V82 + "+dotprod", "dotprod",
         warn=["enable-dotprod without enable-i8mm"]),
    case("fp16", ["-Denable-fp16=true"], V82 + "+fp16+dotprod+i8mm", "i8mm"),
    case("fp16 fp16fml", ["-Denable-fp16=true", "-Denable-fp16fml=true"],
         V82 + "+fp16+dotprod+i8mm+fp16fml", "i8mm"),
    case("fp16 no dotprod", ["-Denable-fp16=true", "-Denable-dotprod=false"],
         V82 + "+fp16+i8mm", "none",
         warn=["enable-fp16 without enable-dotprod",
               "enable-i8mm without enable-dotprod"]),
    case("sve", ["-Denable-sve=true"], DEF + "+sve", "i8mm"),
    case("sve2 implies sve", ["-Denable-sve2=true"], DEF + "+sve+sve2",
         "i8mm"),
    case("fp16-impl ignored", ["-Denable-fp16=true", "-Dfp16-impl=wrapper"],
         V82 + "+fp16+dotprod+i8mm", "i8mm",
         warn=["fp16-impl=wrapper is ignored on this build"]),
    case("fp16fml without fp16", ["-Denable-fp16fml=true"],
         error="enable-fp16fml requires enable-fp16"),
    case("armv8-a fp16", ["-Darm-arch=armv8-a", "-Denable-fp16=true"],
         error="enable-fp16 requires arm-arch=armv8.2-a"),
    case("armv7-a on aarch64", ["-Darm-arch=armv7-a"],
         error="arm-arch=armv7-a is not supported on aarch64"),
]

AARCH64_NOSME = [
    case("sme rejected by compiler", ["-Denable-sme=true"],
         error="does not support " + DEF + "+sme"),
]

ARMHF = [
    case("default", [], "-march=armv7-a", "none"),
    case("fp16", ["-Denable-fp16=true"], "-march=armv7-a", "none",
         warn=["FP16 is enabled on armv7"]),
    case("armv8-a on arm", ["-Darm-arch=armv8-a"],
         error="arm-arch=armv8-a is not supported on 32-bit arm"),
    case("no neon", ["-Denable-neon=false"],
         error="armv7-a requires enable-neon"),
    case("sve ignored", ["-Denable-sve=true"], "-march=armv7-a", "none",
         warn=["enable-sve is ignored on non-aarch64 builds"]),
]

X86 = [
    case("default", [], kai="none"),
    case("arm-arch ignored", ["-Darm-arch=armv8-a"], kai="none",
         warn=["arm-arch=armv8-a is ignored on non-ARM builds"]),
    case("sme ignored", ["-Denable-sme=true"], kai="none",
         warn=["enable-sme is ignored on non-aarch64 builds"]),
]

ANDROID = [
    case("default", [], DEF, "i8mm"),
    case("armv8-a plain",
         ["-Darm-arch=armv8-a", "-Denable-dotprod=false",
          "-Denable-i8mm=false"], "-march=armv8-a", "none"),
    case("no i8mm", ["-Denable-i8mm=false"], V82 + "+dotprod", "dotprod",
         warn=["enable-dotprod without enable-i8mm"]),
    case("fp16", ["-Denable-fp16=true"], V82 + "+fp16+dotprod+i8mm", "i8mm"),
    case("sve2 sme2 implications",
         ["-Denable-sve2=true", "-Denable-sme2=true"],
         DEF + "+sve+sve2+sme+sme2", "i8mm"),
    case("fp16fml without fp16", ["-Denable-fp16fml=true"],
         error="enable-fp16fml requires enable-fp16"),
    case("NDK clang rejects -march", [], ndk="reject",
         error="NDK clang does not support " + DEF),
    case("NDK clang missing", [], DEF, "i8mm", ndk="noclang",
         warn=["NDK clang not found"]),
]

SUITES = {
    "aarch64": AARCH64,
    "aarch64-nosme": AARCH64_NOSME,
    "armhf": ARMHF,
    "x86": X86,
    "android": ANDROID,
}

CROSS_CPU = {
    "aarch64": ("aarch64", "armv8-a"),
    "aarch64-nosme": ("aarch64", "armv8-a"),
    "armhf": ("arm", "armv7"),
}


def write_cross_file(path, prefix, suite):
    family, cpu = CROSS_CPU[suite]
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "[binaries]\n"
            f"c = '{prefix}gcc'\n"
            f"cpp = '{prefix}g++'\n"
            f"ar = '{prefix}ar'\n"
            f"strip = '{prefix}strip'\n"
            "[host_machine]\n"
            "system = 'linux'\n"
            f"cpu_family = '{family}'\n"
            f"cpu = '{cpu}'\n"
            "endian = 'little'\n")


def make_fake_ndk(root, mode):
    """An ndk-build that is never run, with a clang that always fails or none"""
    os.makedirs(root)
    ndk_build = os.path.join(root, "ndk-build")
    with open(ndk_build, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nexit 1\n")
    os.chmod(ndk_build, 0o755)
    if mode == "reject":
        bindir = os.path.join(root, "toolchains", "llvm", "prebuilt",
                              "linux-x86_64", "bin")
        os.makedirs(bindir)
        clang = os.path.join(bindir, "clang")
        with open(clang, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho 'clang: error: fake reject' >&2\n"
                    "exit 1\n")
        os.chmod(clang, 0o755)


def nntrainer_commands(builddir, srcdir):
    """(file, command) of every non-subproject compile command"""
    with open(os.path.join(builddir, "compile_commands.json"),
              encoding="utf-8") as f:
        entries = json.load(f)
    subprojects = os.path.join(srcdir, "subprojects") + os.sep
    out = []
    for e in entries:
        path = os.path.normpath(os.path.join(e["directory"], e["file"]))
        if not path.startswith(subprojects):
            out.append((path, e.get("command") or " ".join(e["arguments"])))
    return out


def android_commands(builddir):
    """(sources, flag lines) of the generated Android.mk"""
    with open(os.path.join(builddir, "jni", "Android.mk"),
              encoding="utf-8") as f:
        text = f.read()
    flags = re.findall(r"^LOCAL_C(?:XX)?FLAGS\s*[:+]?=(.*)$", text, re.M)
    srcs = []
    for line in re.findall(r"^LOCAL_SRC_FILES\s*[:+]?=(.*)$", text, re.M):
        srcs += line.split()
    return srcs, flags


def check_flags(c, suite, lines):
    """lines: compile command strings that must all carry the ISA flags"""
    if not lines:
        return ["no compile flags found to check"]
    errs = []
    for cmd in lines:
        marchs = re.findall(r"-march=\S+", cmd)
        want = [c["march"]] if c["march"] else []
        if marchs != want:
            errs.append(f"-march {marchs} != {want} in: {cmd[:160]}")
            break
        if suite == "armhf" and "-mfpu=neon" not in cmd.split():
            errs.append(f"-mfpu=neon missing in: {cmd[:160]}")
            break
        if "-D__ARM_FEATURE_" in cmd:
            errs.append(f"forced __ARM_FEATURE_* define in: {cmd[:160]}")
            break
    return errs


def check_kai(c, files):
    if c["kai"] is None:
        return []
    if not files:
        return ["no source files found to check"]
    dotprod = any(f.endswith("_neon_dotprod.c") for f in files)
    i8mm = any(f.endswith("_neon_i8mm.c") for f in files)
    want = {"none": (False, False), "dotprod": (True, False),
            "i8mm": (True, True)}[c["kai"]]
    if (dotprod, i8mm) != want:
        return [f"KleidiAI (dotprod, i8mm) sources {(dotprod, i8mm)} != {want}"]
    return []


def run_case(c, suite, srcdir, workdir, cross_file, blas_cache):
    builddir = os.path.join(workdir, "build")
    # Android configure downloads OpenBLAS into each builddir unless present
    if suite == "android" and os.path.isdir(blas_cache):
        shutil.copytree(blas_cache, os.path.join(builddir, "openblas"))
    env = dict(os.environ)
    if c["ndk"]:
        fake = os.path.join(workdir, "fake-ndk")
        make_fake_ndk(fake, c["ndk"])
        env["PATH"] = fake + os.pathsep + env["PATH"]
    cmd = ["meson", "setup", builddir, srcdir] + COMMON_OPTS
    cmd += ANDROID_OPTS if suite == "android" else LINUX_OPTS
    cmd += c["opts"]
    if cross_file:
        cmd += ["--cross-file", cross_file]
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, check=False)
    log = p.stdout
    errs = []

    if c["error"]:
        if p.returncode == 0:
            errs.append(f"configure passed, expected error '{c['error']}'")
        elif c["error"] not in log:
            errs.append(f"error '{c['error']}' not in configure output")
        return errs, log
    if p.returncode != 0:
        return [f"configure failed (rc={p.returncode})"], log
    blas = os.path.join(builddir, "openblas")
    if suite == "android" and os.path.isdir(blas) and \
            not os.path.isdir(blas_cache):
        shutil.copytree(blas, blas_cache)

    isa_lines = [l for l in log.splitlines() if "WARNING:" in l and
                 any(w in l for w in ISA_WARNINGS)]
    for w in c["warn"]:
        if not any(w in l for l in isa_lines):
            errs.append(f"missing warning '{w}'")
    for l in isa_lines:
        if not any(w in l for w in c["warn"]):
            errs.append(f"unexpected warning: {l.strip()}")

    if suite == "android":
        srcs, flags = android_commands(builddir)
        errs += check_flags(c, suite, flags)
        errs += check_kai(c, srcs)
    else:
        cmds = nntrainer_commands(builddir, srcdir)
        errs += check_flags(c, suite, [cmd for _, cmd in cmds])
        errs += check_kai(c, [f for f, _ in cmds])
    return errs, log


def main():
    parser = argparse.ArgumentParser(
        description="Check the meson configure results of ARM ISA option "
        "combinations")
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--cross-prefix", default="",
                        help="toolchain prefix, e.g. aarch64-linux-gnu-")
    parser.add_argument("--cross-file",
                        help="meson cross file (overrides --cross-prefix)")
    parser.add_argument("--srcdir", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    parser.add_argument("--keep", action="store_true",
                        help="keep the build directories of failed cases")
    args = parser.parse_args()
    srcdir = os.path.realpath(args.srcdir)
    cross_file = args.cross_file and os.path.realpath(args.cross_file)
    needs_cross = not cross_file and args.suite in CROSS_CPU
    if needs_cross and not args.cross_prefix:
        parser.error(f"--suite {args.suite} needs --cross-prefix "
                     "or --cross-file")

    root = tempfile.mkdtemp(prefix="arm_isa_config_")
    blas_cache = os.path.join(root, "openblas")
    if needs_cross:
        cross_file = os.path.join(root, "cross.ini")
        write_cross_file(cross_file, args.cross_prefix, args.suite)

    failed = 0
    for i, c in enumerate(SUITES[args.suite]):
        workdir = os.path.join(root, f"case{i}")
        os.makedirs(workdir)
        errs, log = run_case(c, args.suite, srcdir, workdir, cross_file,
                             blas_cache)
        if errs:
            failed += 1
            print(f"FAIL [{args.suite}] {c['name']}: {' '.join(c['opts'])}")
            for e in errs:
                print(f"    {e}")
            print("    --- configure output (last 30 lines) ---")
            for line in log.splitlines()[-30:]:
                print(f"    | {line}")
            if args.keep:
                print(f"    kept: {workdir}")
                continue
        else:
            print(f"ok   [{args.suite}] {c['name']}")
        shutil.rmtree(workdir)

    total = len(SUITES[args.suite])
    print(f"{total - failed}/{total} passed")
    if not args.keep or failed == 0:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
