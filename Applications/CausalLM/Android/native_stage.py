#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a staged native variant directory and write its manifest.

A "stage" is one flat directory holding exactly the files a variant ships
(.so libraries and, for device testing, the native executables):

  public stage    what build_android.sh --stage produces: nntrainer's variant
                  output only, no model-extension plugin;
  internal stage  a copy of a public stage plus libqai_ext_model.so, assembled
                  by a downstream build into its own directory.

Two sub-commands:

  check     fail loudly on residue (files of another variant, the SDK FastRPC
            stub, a second libc++_shared.so, a missing runtime dependency, ...)
            and on any drift between the stage and its manifest;
  manifest  run the same check, then write <manifest>: source SHAs, toolchain,
            variant, the ABI defines nntrainer wrote for prebuilt consumers
            (nntrainer-abi.ini), the engine build mode, a record per file
            (sha256, DT_NEEDED, Hexagon arch, page alignment) and the
            model-extension ABI version.

The per-library record has the shape verify_native_package.py writes for an
AAR/APK ({"sha256", "needed" | "dsp_arch"}), so both reports can be compared.
No timestamps or absolute build paths other than SDK roots are recorded: the
manifest of two builds of the same inputs is byte-identical.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_native_package import (  # noqa: E402
    SYSTEM_LIBRARIES, hexagon_arch, readelf_needed)

SCHEMA = 1
EM_AARCH64 = 183
EM_HEXAGON = 164
PAGE_16K = 0x4000
# Written by the script that created the directory, so it can be recreated
# safely; not part of what the variant ships.
STAGE_MARKER = ".nntr-stage"

# The libraries every variant carries: the engine, its C++ API, the CausalLM
# host and the C API.
CORE_LIBRARIES = ("libnntrainer.so", "libccapi-nntrainer.so", "libcausallm.so",
                  "libquick_dot_ai_api.so")
QNN_REQUIRED = ("libqnn_context.so", "libQnnHtp.so",
                "libQnnHtpNetRunExtensions.so", "libQnnHtpPrepare.so",
                "libQnnSystem.so")
HTP_REQUIRED = ("libsdkl.so", "libnntr_hvx_skel.so")
PLUGIN = "libqai_ext_model.so"
PLUGIN_EXPORTS = ("qai_ext_model_abi_info", "qai_ext_model_register")
# Retired by the consolidation: the legacy product library that used to carry
# the common code. It must not reappear in any stage.
RETIRED = ("libquick_dot_ai.so", "libcausallm_core.so")


def is_qnn_file(name):
    return name == "libqnn_context.so" or name.startswith("libQnn")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def elf_machine(path):
    with open(path, "rb") as handle:
        head = handle.read(20)
    if head[:4] != b"\x7fELF" or len(head) < 20:
        return None
    return struct.unpack_from("<H", head, 18)[0]


def load_alignment(readelf, path):
    """Largest p_align of the PT_LOAD segments (16 KiB-page readiness)."""
    out = subprocess.check_output([readelf, "-lW", str(path)], text=True)
    aligns = []
    for line in out.splitlines():
        fields = line.split()
        if fields and fields[0] == "LOAD":
            aligns.append(int(fields[-1], 16))
    return max(aligns) if aligns else None


def dynamic_symbols(readelf, path):
    out = subprocess.check_output([readelf, "--dyn-syms", "-W", str(path)],
                                  text=True)
    names = set()
    for line in out.splitlines():
        fields = line.split()
        # Num: Value Size Type Bind Vis Ndx Name -- defined symbols only.
        if len(fields) >= 8 and fields[6] != "UND":
            names.add(fields[7].split("@")[0])
    return names


def parse_ini_value(text, key):
    """Value of `key = <python literal>` in a meson machine-file section."""
    match = re.search(r"^%s\s*=\s*(.+)$" % re.escape(key), text, re.M)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1).strip())
    except (ValueError, SyntaxError):
        return match.group(1).strip()


def read_mode_file(path):
    """causallm-build-mode.ini: schema=2 / variant=cpu / engine_arg=... lines."""
    mode = {}
    if not path or not Path(path).is_file():
        return mode
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key == "engine_arg":
            mode.setdefault("engine_args", []).append(value)
        else:
            mode[key] = value
    return mode


def git_info(path):
    path = str(path)

    def run(*args):
        return subprocess.run(["git", "-C", path, *args], text=True,
                              capture_output=True)

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"path": path, "sha": None}
    status = run("status", "--porcelain", "--untracked-files=no")
    subject = run("log", "-1", "--format=%s")
    return {
        "path": path,
        "sha": head.stdout.strip(),
        "subject": subject.stdout.strip(),
        # Tracked files modified relative to HEAD: the SHA alone does not
        # reproduce this build.
        "dirty": bool(status.stdout.strip()),
    }


def tool_version(*cmd):
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else None


def toolchain(variant, mode, nntrainer_root):
    ndk = os.environ.get("ANDROID_NDK") or os.environ.get("NDK_ROOT")
    info = {"meson": tool_version("meson", "--version"),
            "ninja": tool_version("ninja", "--version")}
    if ndk:
        props = Path(ndk) / "source.properties"
        revision = None
        if props.is_file():
            revision = parse_ini_value(
                "\n".join(l.replace(" =", "=").replace("= ", "=")
                          for l in props.read_text().splitlines()),
                "Pkg.Revision")
        clang = sorted((Path(ndk) / "toolchains/llvm/prebuilt").glob(
            "*/bin/clang"))
        info["android_ndk"] = {
            "path": str(Path(ndk).resolve()),
            "revision": revision,
            "clang": tool_version(str(clang[0]), "--version") if clang else None,
        }
    if variant in ("qnn", "qnn-htp"):
        root = os.environ.get("QNN_SDK_ROOT")
        qnn = {"root": root, "version": None, "dir_name": None}
        if root:
            qnn["dir_name"] = Path(root).resolve().name
            sdk_yaml = Path(root) / "sdk.yaml"
            if sdk_yaml.is_file():
                match = re.search(r"^version:\s*(\S+)", sdk_yaml.read_text(), re.M)
                qnn["version"] = match.group(1) if match else None
        info["qnn_sdk"] = qnn
    if variant in ("htp", "qnn-htp"):
        sdk = os.environ.get("HEXAGON_SDK_ROOT")
        kl = os.environ.get("HEXKL_ROOT")
        tools = os.environ.get("DEFAULT_HEXAGON_TOOLS_ROOT")
        info["hexagon_sdk"] = {
            "root": sdk,
            "dir_name": Path(sdk).resolve().name if sdk else None,
            "tools_root": tools,
            "hex_arch": mode.get("hex_arch") or os.environ.get("HEX_ARCH"),
        }
        info["hexkl"] = {
            "root": kl,
            "sdk_version": mode.get("hexkl_lib", "").split("/")[0]
            or os.environ.get("HEXKL_SDK_VER"),
            "lib_subdir": mode.get("hexkl_lib") or None,
        }
    return info


def extension_abi_version(nntrainer_root):
    header = nntrainer_root / "Applications/CausalLM/api/ext_model_abi.h"
    if not header.is_file():
        return None
    match = re.search(r"#\s*define\s+QDA_EXT_ABI_VERSION\s+(\d+)",
                      header.read_text())
    return int(match.group(1)) if match else None


def describe_stage(stage, readelf):
    """One record per file in the stage, plus what `check` needs."""
    files = {}
    for entry in sorted(stage.iterdir(), key=lambda p: p.name):
        if entry.name == STAGE_MARKER:
            continue
        if entry.is_dir():
            files[entry.name] = {"kind": "directory"}
            continue
        record = {"sha256": sha256_of(entry), "size": entry.stat().st_size}
        machine = elf_machine(entry)
        if machine is None:
            record["kind"] = "other"
        elif machine == EM_HEXAGON:
            record["kind"] = "dsp_image"
            data = entry.read_bytes()
            record["dsp_arch"] = "v%x" % hexagon_arch(data)
        elif machine == EM_AARCH64:
            record["kind"] = ("shared_library" if entry.suffix == ".so"
                              else "executable")
            record["needed"] = readelf_needed(readelf, entry)
            record["load_align"] = load_alignment(readelf, entry)
        else:
            record["kind"] = "foreign_elf"
            record["machine"] = machine
        files[entry.name] = record
    return files


def check_stage(variant, kind, files, stage, readelf):
    """Residue and dependency rules; returns a list of violation strings."""
    problems = []
    names = set(files)

    def need(name, why):
        if name not in names:
            problems.append(f"missing {name}: {why}")

    def forbid(predicate, why):
        for name in sorted(n for n in names if predicate(n)):
            problems.append(f"residue {name}: {why}")

    for name, record in files.items():
        if record["kind"] == "directory":
            problems.append(f"unexpected directory {name}/ in the stage")
        elif record["kind"] == "foreign_elf":
            problems.append(f"{name}: unexpected ELF machine {record['machine']}")
        elif record["kind"] == "other":
            problems.append(f"{name}: not an ELF file")

    # -- every variant --------------------------------------------------------
    forbid(lambda n: n == "libcdsprpc.so",
           "the SDK FastRPC link stub must not be packaged")
    forbid(lambda n: n in RETIRED, "retired legacy library")
    runtimes = sorted(n for n in names if n == "libc++_shared.so")
    if len(runtimes) != 1:
        problems.append(f"expected exactly one libc++_shared.so, found {len(runtimes)}")
    for name in CORE_LIBRARIES:
        need(name, "core library of every variant")

    # -- variant rules ---------------------------------------------------------
    if variant == "cpu":
        forbid(is_qnn_file, "QNN library in a CPU/OpenCL stage")
        forbid(lambda n: n in HTP_REQUIRED, "HTP library in a CPU/OpenCL stage")
    elif variant not in ("qnn", "htp", "qnn-htp"):
        problems.append(f"unknown variant {variant}")
    if variant in ("qnn", "qnn-htp"):
        for name in QNN_REQUIRED:
            need(name, "QNN variant runtime")
        if variant == "qnn":
            forbid(lambda n: n in HTP_REQUIRED, "HTP library in a QNN stage")
        pairs = 0
        for version in (75, 79, 81):
            stub = f"libQnnHtpV{version}Stub.so"
            skel = f"libQnnHtpV{version}Skel.so"
            if stub in names and skel in names:
                pairs += 1
                image = files[skel]
                if image.get("kind") != "dsp_image" or image.get("dsp_arch") != f"v{version}":
                    problems.append(f"{skel} is not a matching v{version} DSP image")
            elif stub in names or skel in names:
                problems.append(f"incomplete QNN HTP V{version} Stub/Skel pair")
        if pairs == 0:
            problems.append("no complete QNN HTP Stub/Skel pair")
        if variant == "qnn-htp":
            for name in ("libQnnHtpV81Stub.so", "libQnnHtpV81Skel.so"):
                need(name, "QNN/HTP combined v81 target")
    if variant in ("htp", "qnn-htp"):
        for name in HTP_REQUIRED:
            need(name, "native HTP runtime")
        if variant == "htp":
            forbid(is_qnn_file, "QNN library in a native HTP stage")
        skel = files.get("libnntr_hvx_skel.so")
        if skel and skel.get("kind") != "dsp_image":
            problems.append("libnntr_hvx_skel.so is not a Hexagon DSP image")
        elif skel and skel.get("dsp_arch") != "v81":
            problems.append(f"libnntr_hvx_skel.so targets {skel.get('dsp_arch')},"
                            " expected v81")

    # -- public / internal -----------------------------------------------------
    if kind == "public":
        forbid(lambda n: n == PLUGIN,
               "the model-extension plugin does not belong in the public stage")
    elif kind == "internal":
        need(PLUGIN, "the internal stage adds the model-extension plugin")
        if PLUGIN in names:
            exported = dynamic_symbols(readelf, stage / PLUGIN)
            for symbol in PLUGIN_EXPORTS:
                if symbol not in exported:
                    problems.append(f"{PLUGIN} does not export {symbol}")

    # -- runtime dependencies --------------------------------------------------
    for name, record in files.items():
        for dep in record.get("needed", []):
            if dep not in names and dep not in SYSTEM_LIBRARIES:
                problems.append(f"{name} needs {dep}, which is neither staged"
                                " nor a platform library")
    return problems


def compare_with_manifest(files, manifest_path):
    """The stage must be exactly what its manifest says (no stale files)."""
    problems = []
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError) as err:
        return [f"cannot read manifest {manifest_path}: {err}"]
    recorded = {**manifest.get("libraries", {}), **manifest.get("executables", {}),
                **manifest.get("dsp_images", {})}
    for name in sorted(set(files) - set(recorded)):
        problems.append(f"stale {name}: in the stage but not in the manifest")
    for name in sorted(set(recorded) - set(files)):
        problems.append(f"missing {name}: in the manifest but not in the stage")
    for name in sorted(set(files) & set(recorded)):
        if files[name].get("sha256") != recorded[name].get("sha256"):
            problems.append(f"{name}: sha256 differs from the manifest")
    return problems


def build_manifest(args, files, mode, abi_text):
    nntrainer_root = Path(args.nntrainer_root).resolve()
    abi_cflags = parse_ini_value(abi_text, "nntrainer_abi_cflags") or []
    abi = {
        "schema": parse_ini_value(abi_text, "nntrainer_abi_schema"),
        "cflags": abi_cflags,
        "arm_march": parse_ini_value(abi_text, "nntrainer_arm_march"),
        "enable_fp16": parse_ini_value(abi_text, "nntrainer_enable_fp16"),
        "enable_opencl": "-DENABLE_OPENCL=1" in abi_cflags,
        "enable_hexkl": "-DENABLE_HEXKL=1" in abi_cflags,
        "enable_qnn": args.variant in ("qnn", "qnn-htp"),
    }
    sources = {"nntrainer": git_info(nntrainer_root)}
    for item in args.source:
        name, _, path = item.partition("=")
        sources[name] = git_info(path)
    libraries, executables, dsp_images = {}, {}, {}
    for name, record in files.items():
        entry = {k: v for k, v in record.items() if k != "kind"}
        entry["kind"] = record["kind"]
        if record["kind"] == "shared_library":
            libraries[name] = entry
        elif record["kind"] == "executable":
            executables[name] = entry
        elif record["kind"] == "dsp_image":
            dsp_images[name] = entry
        else:
            libraries[name] = entry
    manifest = {
        "schema": SCHEMA,
        "kind": args.kind,
        "variant": args.variant,
        "sources": sources,
        "toolchain": toolchain(args.variant, mode, nntrainer_root),
        "abi": abi,
        "engine_build_mode": mode,
        "extension_abi_version": extension_abi_version(nntrainer_root),
        "libraries": libraries,
        "executables": executables,
        "dsp_images": dsp_images,
    }
    if args.variant in ("htp", "qnn-htp"):
        generated = nntrainer_root / "nntrainer/tensor/htp_backend/generated"
        idl = nntrainer_root / "test/htp/nntr_hvx.idl"
        stub = {}
        for path in (idl, generated / "nntr_hvx.h", generated / "nntr_hvx_stub.c"):
            if path.is_file():
                stub[path.name] = sha256_of(path)
        # The ARM half (client stub, compiled into libnntrainer.so) and the DSP
        # half (the skel in dsp_images) are generated from the one IDL.
        manifest["htp_interface"] = {"stub_and_idl_sha256": stub,
                                     "skel": sorted(dsp_images)}
    if args.kind == "internal":
        manifest["plugin"] = {"library": PLUGIN,
                              "sha256": files.get(PLUGIN, {}).get("sha256")}
    if args.base_manifest:
        base = Path(args.base_manifest)
        manifest["derived_from"] = {"public_manifest": base.name,
                                    "sha256": sha256_of(base)}
    manifest["alignment_16k"] = {
        name: record["load_align"] >= PAGE_16K
        for name, record in sorted(libraries.items())
        if record.get("load_align") is not None}
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "manifest"))
    parser.add_argument("--variant", choices=("cpu", "qnn", "htp", "qnn-htp"), required=True)
    parser.add_argument("--kind", choices=("public", "internal"), default="public")
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--nntrainer-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--engine-result", type=Path,
                        help="engine install dir holding nntrainer-abi.ini and "
                             "causallm-build-mode.ini (default: "
                             "<nntrainer>/builddir/android_build_result)")
    parser.add_argument("--abi-file", type=Path)
    parser.add_argument("--mode-file", type=Path)
    parser.add_argument("--source", action="append", default=[],
                        metavar="NAME=GIT_DIR",
                        help="extra source tree to record (nntrainer is always recorded)")
    parser.add_argument("--base-manifest", type=Path,
                        help="manifest of the public stage an internal stage is copied from")
    parser.add_argument("--manifest", type=Path,
                        help="check: also require the stage to match this manifest")
    parser.add_argument("--output", type=Path, help="manifest: file to write")
    args = parser.parse_args(argv)

    stage = args.stage_dir
    if not stage.is_dir():
        sys.exit(f"error: stage directory not found: {stage}")

    problems = []
    if args.command == "check" and args.manifest:
        # Checking a stage that was built earlier: what it was built from is in
        # its manifest, and the live engine directory may since have been
        # rebuilt for another variant.
        try:
            recorded = json.loads(args.manifest.read_text())
        except (OSError, ValueError) as err:
            sys.exit(f"error: cannot read manifest {args.manifest}: {err}")
        abi_cflags = recorded.get("abi", {}).get("cflags", [])
        mode = recorded.get("engine_build_mode", {})
        abi_text = None
    else:
        engine = args.engine_result or (
            args.nntrainer_root / "builddir/android_build_result")
        abi_file = args.abi_file or engine / "nntrainer-abi.ini"
        mode_file = args.mode_file or engine / "causallm-build-mode.ini"
        abi_text = abi_file.read_text() if abi_file.is_file() else ""
        mode = read_mode_file(mode_file)
        abi_cflags = parse_ini_value(abi_text, "nntrainer_abi_cflags") or []
        if not abi_text:
            problems.append(f"missing ABI metadata {abi_file}")
        if not mode:
            problems.append(f"missing engine build mode {mode_file}")

    files = describe_stage(stage, args.readelf)
    problems += check_stage(args.variant, args.kind, files, stage, args.readelf)

    # The engine metadata the build consumed must describe THIS variant: a stage
    # assembled from an engine directory left by another variant is the stale
    # mix this tool exists to catch.
    if mode.get("variant") not in (None, args.variant):
        problems.append(f"engine build mode is {mode.get('variant')}, "
                        f"stage is {args.variant}")
    if mode.get("enable_opencl") in ("true", "false") and \
            (mode["enable_opencl"] == "true") != ("-DENABLE_OPENCL=1" in abi_cflags):
        problems.append("nntrainer-abi.ini ENABLE_OPENCL does not match the "
                        f"engine build mode (enable_opencl={mode['enable_opencl']})")
    if ("-DENABLE_HEXKL=1" in abi_cflags) != (args.variant in ("htp", "qnn-htp")):
        problems.append("nntrainer-abi.ini ENABLE_HEXKL does not match variant "
                        f"{args.variant}")
    if args.manifest:
        problems += compare_with_manifest(files, args.manifest)

    if problems:
        print(f"FAIL: {args.kind} {args.variant} stage {stage}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if args.command == "check":
        print(f"OK: {args.kind} {args.variant} stage {stage}: {len(files)} files, "
              "no residue")
        return 0

    if not args.output:
        sys.exit("error: manifest needs --output")
    manifest = build_manifest(args, files, mode, abi_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"OK: {args.kind} {args.variant} stage {stage}: {len(files)} files, "
          f"no residue; manifest {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
