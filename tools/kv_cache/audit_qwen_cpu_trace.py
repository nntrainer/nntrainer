#!/usr/bin/env python3
"""Check the CPU trace witnesses against the declared Qwen cache allocations.

This checks instrumented application operations, not hardware memory traffic.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def audit(path):
    text = path.read_text()
    header_pattern = re.compile(r"\[kv-cpu-storage\] backing_bytes=(\d+) owner_layers=(\d+)")
    allocation_pattern = re.compile(
        r"\[kv-cpu-storage\] layer=(\d+) key=(0x[0-9a-f]+) value=(0x[0-9a-f]+) "
        r"plane_bytes=(\d+) source=(-?\d+)")
    access_pattern = re.compile(
        r"\[kv-cpu-access\] from=(\d+) tokens=(\d+) key=(0x[0-9a-f]+) "
        r"value=(0x[0-9a-f]+) read_alias=(\d+) write_alias=(\d+) "
        r"append_copy_bytes=(\d+) transform_bytes=(\d+)")
    instances, pair_instances, accesses = [], {}, []
    groups = defaultdict(Counter)
    # Save/resume keeps the original and restored models alive together. Resolve
    # each access to the allocation instance that actually supplied its pair.
    for line in text.splitlines():
        header = header_pattern.search(line)
        allocation = allocation_pattern.search(line)
        access = access_pattern.search(line)
        if header:
            size, owners = map(int, header.groups())
            instances.append({"backing_bytes": size, "owner_layers": owners, "pairs": {}})
        elif allocation:
            if not instances:
                raise ValueError(f"Storage plane without an owner header in {path}")
            _, key, value, size, source = allocation.groups()
            pair_instances[(key, value)] = len(instances) - 1
            if int(source) == -1:
                instances[-1]["pairs"][(key, value)] = int(size)
        elif access:
            row = access.groups()
            position, tokens, key, value, read_alias, write_alias, copy, transform = row
            if (key, value) not in pair_instances or read_alias != "1" or write_alias != "1" or copy != "0":
                raise ValueError(f"Cache alias or append copy mismatch in {path}")
            if int(position) + int(tokens) > 512 or int(transform) != int(tokens) * 1024 * 2 * 2:
                raise ValueError(f"Unexpected Qwen geometry or transform byte count in {path}")
            owner = pair_instances[(key, value)]
            groups[(owner, position, tokens)][(key, value)] += 1
            accesses.append(row)
    if not instances:
        raise ValueError(f"No core CPU storage witness in {path}")
    for instance in instances:
        owners = instance["pairs"]
        if len(owners) != instance["owner_layers"] or sum(owners.values()) * 2 != instance["backing_bytes"]:
            raise ValueError(f"Storage byte or owner count mismatch in {path}")
    reads = re.findall(
        r"\[kv-cpu-read\] consumer=([^ ]+) history_copy_bytes=(\d+)([^\n]*)", text)
    if not accesses or len(reads) != len(accesses):
        raise ValueError(f"Missing producer or consumer witnesses in {path}")
    for (owner, _, _), group in groups.items():
        if set(group) != set(instances[owner]["pairs"]) or any(count != 1 for count in group.values()):
            raise ValueError(f"A step is missing or duplicating a KV owner in {path}")
    if "profile=compatibility" in text or re.search(
            r"\[kv-cpu-layout\].*history_copy_bytes=[1-9][0-9]*", text):
        raise ValueError(f"A compatibility or layout history-copy path ran in {path}")
    if any(copy != "0" or arm not in ("flash-strided", "per-row", "flash-tiled")
           for arm, copy, _ in reads):
        raise ValueError(f"Unsupported consumer or history copy in {path}")
    tile_pack_bytes, tile_scratch_bytes = 0, 0
    for consumer, _, counters in reads:
        if consumer != "flash-tiled":
            continue
        packed = re.search(r"\btile_pack_bytes=(\d+)\b", counters)
        scratch = re.search(r"\btile_scratch_bytes=(\d+)\b", counters)
        # Qwen head width=128, key tile<=128, two half-bit planes per worker.
        # This is a bounded kernel scratch copy, separate from a full-prefix
        # cache materialization. Do not describe the tile path as copy-free.
        if (not packed or not scratch or int(packed[1]) < 1 or
                not 0 < int(scratch[1]) <= 2 * 128 * 128 * 2):
            raise ValueError(f"Missing or unbounded ARM tile scratch witness in {path}")
        tile_pack_bytes += int(packed[1])
        tile_scratch_bytes = max(tile_scratch_bytes, int(scratch[1]))
    return {"file": str(path), "storage_instances": len(instances),
            "storage_geometry": [{"backing_bytes": i["backing_bytes"],
                                  "owner_layers": i["owner_layers"]} for i in instances],
            "backing_bytes": sum(i["backing_bytes"] for i in instances),
            "owner_layers": sum(i["owner_layers"] for i in instances),
            "producer_calls": len(accesses),
            "consumers": sorted({arm for arm, _, _ in reads}),
            "allocation_identity_matches": True,
            "read_alias": True, "write_alias": True,
            "append_copy_bytes": 0, "history_copy_bytes": 0,
            "tile_pack_bytes": tile_pack_bytes,
            "max_tile_scratch_bytes_per_worker": tile_scratch_bytes,
            "transform_bytes": sum(int(row[-1]) for row in accesses), "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.stderr.txt"))
    report = {"profile": "Qwen3-0.6B, capacity512, kv_width1024, half-bit storage",
              "runs": [audit(path) for path in files]}
    if not files:
        parser.error("No probe stderr traces found")
    report["passed"] = all(row["passed"] for row in report["runs"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(files)} CPU trace files")


if __name__ == "__main__":
    main()
