#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check staged runtime dependencies and preserve the Hexagon image in APK/AAR."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zipfile


# Libraries the Android platform provides; a DT_NEEDED naming one of them is not
# a packaging gap. libcdsprpc.so is listed because the SDK link stub is
# deliberately NOT shipped (the device's own FastRPC library is used).
SYSTEM_LIBRARIES = {
    "libc.so", "libm.so", "libdl.so", "liblog.so", "libandroid.so",
    "libjnigraphics.so", "libz.so", "libEGL.so", "libGLESv2.so",
    "libGLESv3.so", "libOpenCL.so", "libvndksupport.so", "libcdsprpc.so",
}


def readelf_needed(readelf, path):
    """DT_NEEDED names of one AArch64 ELF, in link order."""
    dynamic = subprocess.check_output([readelf, "-d", str(path)], text=True)
    return re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)


def hexagon_arch(data):
    """Hexagon ISA revision (0x81 for v81) from an ELF32 DSP image's e_flags."""
    return struct.unpack_from("<I", data, 36)[0] & 0xFF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "qnn", "htp", "qnn-htp"), required=True)
    parser.add_argument("--prebuilt-dir", type=Path, required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    # Share stage rules so packages require both runtimes for qnn-htp and
    # retain the single-backend residue guards.
    from native_stage import check_stage, describe_stage, PLUGIN
    staged = {p.name: p for p in args.prebuilt_dir.glob("*.so")}
    kind = "internal" if PLUGIN in staged else "public"
    problems = check_stage(args.variant, kind,
                           describe_stage(args.prebuilt_dir, args.readelf),
                           args.prebuilt_dir, args.readelf)
    assert not problems, "Invalid native stage: " + "; ".join(problems)
    assert "libcdsprpc.so" not in staged, "Do not package the SDK FastRPC link stub"
    skel = "libnntr_hvx_skel.so"
    if args.variant in ("htp", "qnn-htp"):
        assert {skel, "libsdkl.so"} <= staged.keys(), "HTP runtime is incomplete"
        if args.variant == "htp":
            assert not any(n.startswith("libQnn") or n == "libqnn_context.so" for n in staged)
    else:
        assert skel not in staged and "libsdkl.so" not in staged, "Stale HTP libraries"

    system = SYSTEM_LIBRARIES
    report = {"variant": args.variant, "packages": []}
    for package in args.packages:
        prefix = "jni/arm64-v8a/" if package.suffix == ".aar" else "lib/arm64-v8a/"
        entry = {"path": str(package), "libraries": {}}
        with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as tmp:
            names = {Path(n).name for n in archive.namelist() if n.startswith(prefix)}
            assert staged.keys() <= names, f"Missing libraries: {staged.keys() - names}"
            assert "libcdsprpc.so" not in names, "Packaged SDK FastRPC stub"
            records = {}
            for name in sorted(n for n in names if n.endswith(".so")):
                data = archive.read(prefix + name)
                digest = hashlib.sha256(data).hexdigest()
                info = {"sha256": digest}
                machine = struct.unpack_from("<H", data, 18)[0]
                if machine == 164:
                    assert name in staged, f"Unexpected DSP image: {name}"
                    assert data == staged[name].read_bytes(), f"DSP image changed in {package}"
                    assert data[:6] == b"\x7fELF\x01\x01", "Expected ELF32 little endian"
                    arch = hexagon_arch(data)
                    if name == skel:
                        assert arch == 0x81, "Expected v81"
                    info["dsp_arch"] = f"v{arch:x}"
                    records[name] = dict(info, kind="dsp_image")
                else:
                    assert machine == 183, f"Expected AArch64: {name}"
                    assert name != skel, "Expected Hexagon DSP image"
                    path = Path(tmp) / name
                    path.write_bytes(data)
                    needed = readelf_needed(args.readelf, path)
                    missing = set(needed) - names - system
                    assert not missing, f"{name} has missing dependencies: {missing}"
                    info["needed"] = needed
                    records[name] = dict(info, kind="shared_library")
                entry["libraries"][name] = info
            problems = check_stage(args.variant, kind, records,
                                   args.prebuilt_dir, args.readelf)
            assert not problems, f"Invalid package {package}: " + "; ".join(problems)
        report["packages"].append(entry)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {args.variant} native packaging: {args.output}")


if __name__ == "__main__":
    main()
