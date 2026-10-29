#!/usr/bin/env python3
"""Freeze only Meson's current shared-library targets, excluding old snapshots.

Recursive filename searches can mix preserved baseline libraries into a new
runtime. Read the build graph instead, reject basename collisions, and record
every source path and hash. Use a new output directory for each revision.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    """Hash a library without loading the complete binary into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(build, output):
    """Copy stable Meson ELF library targets into a fresh immutable snapshot."""
    targets = build / "meson-info" / "intro-targets.json"
    selected = {}
    for target in json.loads(targets.read_text(encoding="utf-8")):
        if target["type"] != "shared library":
            continue
        for name in target["filename"]:
            source = Path(name)
            if not source.is_absolute():
                source = build / source
            if ".so" not in source.name:
                continue
            if not source.is_file():
                raise ValueError(f"Missing built shared library: {source}")
            if source.name in selected and selected[source.name] != source:
                raise ValueError(f"Ambiguous runtime library basename: {source.name}")
            selected[source.name] = source
    if not selected:
        raise ValueError("No ELF shared-library targets in this Meson build")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; snapshots are immutable")
    output.mkdir(parents=True, exist_ok=True)
    libraries = []
    for name, source in sorted(selected.items()):
        before = sha256(source)
        destination = output / name
        shutil.copy2(source, destination)
        if sha256(destination) != before or sha256(source) != before:
            raise ValueError(f"Library changed while freezing: {source}")
        libraries.append({"name": name, "source": str(source.resolve()),
                          "bytes": destination.stat().st_size, "sha256": before})
    report = {"build": str(build.resolve()), "targets_sha256": sha256(targets),
              "selection": "Meson shared-library targets only", "libraries": libraries}
    (output / "runtime-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    """Freeze the explicitly selected build and print the selected target count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze(args.build.resolve(), args.output.resolve())
    print(f"Froze {len(report['libraries'])} shared-library targets")


if __name__ == "__main__":
    main()
