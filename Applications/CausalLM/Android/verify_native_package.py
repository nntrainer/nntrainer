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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "qnn", "htp"), required=True)
    parser.add_argument("--prebuilt-dir", type=Path, required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    staged = {p.name: p for p in args.prebuilt_dir.glob("*.so")}
    assert "libcdsprpc.so" not in staged, "Do not package the SDK FastRPC link stub"
    skel = "libnntr_hvx_skel.so"
    if args.variant == "htp":
        assert {skel, "libsdkl.so"} <= staged.keys(), "HTP runtime is incomplete"
        assert not any(n.startswith("libQnn") or n == "libqnn_context.so" for n in staged)
    else:
        assert skel not in staged and "libsdkl.so" not in staged, "Stale HTP libraries"

    system = {
        "libc.so", "libm.so", "libdl.so", "liblog.so", "libandroid.so",
        "libjnigraphics.so", "libz.so", "libEGL.so", "libGLESv2.so",
        "libGLESv3.so", "libOpenCL.so", "libvndksupport.so", "libcdsprpc.so",
    }
    report = {"variant": args.variant, "packages": []}
    for package in args.packages:
        prefix = "jni/arm64-v8a/" if package.suffix == ".aar" else "lib/arm64-v8a/"
        entry = {"path": str(package), "libraries": {}}
        with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as tmp:
            names = {Path(n).name for n in archive.namelist() if n.startswith(prefix)}
            assert staged.keys() <= names, f"Missing libraries: {staged.keys() - names}"
            assert "libcdsprpc.so" not in names, "Packaged SDK FastRPC stub"
            for name in sorted(staged):
                data = archive.read(prefix + name)
                digest = hashlib.sha256(data).hexdigest()
                info = {"sha256": digest}
                machine = struct.unpack_from("<H", data, 18)[0]
                if machine == 164:
                    assert data == staged[name].read_bytes(), f"DSP image changed in {package}"
                    assert data[:6] == b"\x7fELF\x01\x01", "Expected ELF32 little endian"
                    arch = struct.unpack_from("<I", data, 36)[0] & 0xFF
                    if name == skel:
                        assert arch == 0x81, "Expected v81"
                    info["dsp_arch"] = f"v{arch:x}"
                else:
                    assert machine == 183, f"Expected AArch64: {name}"
                    assert name != skel, "Expected Hexagon DSP image"
                    path = Path(tmp) / name
                    path.write_bytes(data)
                    dynamic = subprocess.check_output([args.readelf, "-d", str(path)], text=True)
                    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
                    missing = set(needed) - names - system
                    assert not missing, f"{name} has missing dependencies: {missing}"
                    info["needed"] = needed
                entry["libraries"][name] = info
        report["packages"].append(entry)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {args.variant} native packaging: {args.output}")


if __name__ == "__main__":
    main()
