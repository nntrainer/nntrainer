#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare rotations in an already deployed CausalLM app, on one explicit device.

The model must use 1024 input tokens and 127 decode iterations. Each run is a
fresh process. Rotating an ordinary checkpoint changes its outputs and routing;
these timings do not establish overhead at identical expert assignments.
"""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile
import time


@contextmanager
def staged_model(adb, root, config):
    """Stage only the run's config; keep the original model pack untouched."""
    remote = subprocess.check_output(
        adb + ["shell", shlex.join(["mktemp", "-d", root + "/results/hadamard-config.XXXXXX"])],
        text=True).strip()
    if not remote.startswith(root + "/results/hadamard-config.") or "\n" in remote:
        raise RuntimeError("Unexpected temporary model directory: " + remote)
    try:
        source = shlex.quote(root + "/model")
        destination = shlex.quote(remote)
        script = (
            f'for entry in {source}/* {source}/.[!.]* {source}/..?*; do '
            '[ -e "$entry" ] || continue; '
            'name=${entry##*/}; [ "$name" = nntr_config.json ] && continue; '
            f'ln -s "$entry" {destination}/"$name" || exit; done'
        )
        subprocess.run(adb + ["shell", script], check=True)
        with tempfile.TemporaryDirectory(prefix="nntr-hadamard-config-") as local:
            path = Path(local) / "nntr_config.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            subprocess.run(adb + ["push", str(path), remote + "/nntr_config.json"],
                           check=True)
        yield remote
    finally:
        subprocess.run(adb + ["shell", shlex.join(["rm", "-rf", remote])], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", default="off,both_hmx,both,both,both_hmx,off")
    parser.add_argument("--profile", action="store_true", help="diagnostic only")
    parser.add_argument("--cache-experts", type=int, default=16,
                        help="expert cache capacity per layer, same for all variants")
    parser.add_argument("--cache-grow", type=int, default=0,
                        help="additional experts per layer after the first decode token")
    parser.add_argument("--miss-readers", type=int, default=4,
                        help="parallel expert-weight readers, same for all variants")
    args = parser.parse_args()
    if args.serial.lower().endswith("lh"):
        parser.error("Devices ending in LH are excluded.")
    if (not 8 <= args.cache_experts <= 128 or
            not 0 <= args.cache_grow <= 128 - args.cache_experts):
        parser.error("cache capacity must be 8..128 experts per layer")
    if not 1 <= args.miss_readers <= 8:
        parser.error("--miss-readers must be 1..8")
    variants = args.order.split(",")
    bits = {"off": 0, "both": 0x0C000000, "both_hmx": 0x2C000000}
    if any(v not in bits for v in variants):
        parser.error("--order accepts off, both, both_hmx")
    args.output.mkdir(parents=True, exist_ok=False)
    adb = ["adb", "-s", args.serial]
    root = args.remote_root.rstrip("/")
    settings = {
        "LD_LIBRARY_PATH": ".",
        "ADSP_LIBRARY_PATH": ".",
    }
    original = subprocess.check_output(
        adb + ["shell", shlex.join(["cat", root + "/model/nntr_config.json"])])
    (args.output / "original-nntr_config.json").write_bytes(original)
    config = json.loads(original)
    config["num_threads"] = 8
    for key in ("moe_cache_experts", "moe_hadamard", "moe_hadamard_scope"):
        config.pop(key, None)
    config.setdefault("htp", {}).update(page_cache=0, profile=int(args.profile))
    config.setdefault("moe", {}).update(
        cache_experts=args.cache_experts, cache_grow=args.cache_grow,
        miss_readers=args.miss_readers, dma_queues=4, hadamard_scope="2bit")
    subprocess.run(adb + ["shell", shlex.join(["mkdir", "-p", root + "/results"])], check=True)
    runtime_files = ("libnntr_hvx_skel.so", "libnntrainer.so", "nntrainer_causallm")
    hashes = subprocess.check_output(adb + ["shell", shlex.join(
        ["sha256sum"] + [root + "/runtime/" + name for name in runtime_files]
    )], text=True)
    runtime_hashes = {Path(line.split()[1]).name: line.split()[0]
                      for line in hashes.splitlines()}
    records = []
    (args.output / "settings.json").write_text(json.dumps({
        "serial": args.serial, "remote_root": root, "environment": settings,
        "runtime_config": {key: config[key] for key in ("htp", "moe")},
        "order": variants, "profile": args.profile,
        "measurement": "actual_rotation",
        "runtime_sha256": runtime_hashes,
    }, indent=2) + "\n")
    for index, variant in enumerate(variants):
        tag = f"{index:02d}-{variant}"
        ids_path = f"{root}/results/{tag}.ids"
        env = dict(settings, NNTR_GENERATED_IDS=ids_path)
        run_config = copy.deepcopy(config)
        run_config["moe"]["hadamard"] = variant
        if args.profile:
            run_config["moe"]["trace"] = f"{root}/results/{tag}.route.txt"
            run_config["moe"]["route_log"] = f"{root}/results/{tag}.decode-route.txt"
        (args.output / (tag + ".nntr_config.json")).write_text(
            json.dumps(run_config, indent=2) + "\n")
        with staged_model(adb, root, run_config) as model:
            command = "cd " + shlex.quote(root + "/runtime") + " && exec "
            command += shlex.join(["timeout", "180", "env"] +
                                  [f"{k}={v}" for k, v in env.items()] +
                                  ["./nntrainer_causallm", model])
            (args.output / (tag + ".command.txt")).write_text(command + "\n")
            thermal = subprocess.check_output(adb + ["shell", "cat",
                "/sys/class/thermal/thermal_zone0/temp"], text=True).strip()
            start = time.monotonic()
            with (args.output / (tag + ".log")).open("w") as log:
                subprocess.run(adb + ["shell", command], stdout=log,
                               stderr=subprocess.STDOUT, timeout=210, check=True)
            elapsed = time.monotonic() - start
        log = (args.output / (tag + ".log")).read_text()
        prefill = re.search(r"prefill: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", log)
        decode = re.search(r"generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", log)
        inference = re.search(r"^total: (\d+) ms", log, re.MULTILINE)
        e2e = re.search(r"\[e2e time\]: (\d+) ms", log)
        flags = re.search(r"moe m1 gemv: on \(applied=0x([0-9a-f]+)\)", log)
        if not all((prefill, decode, inference, e2e, flags)):
            raise RuntimeError(f"Incomplete run: {tag}; inspect its log")
        if (int(prefill[1]), int(decode[1])) != (1024, 127):
            raise RuntimeError(f"Wrong token counts: {tag}")
        scope = 0x40000000 if bits[variant] else 0
        if int(flags[1], 16) != (0x00CF03E1 | bits[variant] | scope):
            raise RuntimeError(f"Wrong DSP flags: {tag}")
        ids = subprocess.check_output(adb + ["shell", shlex.join(["cat", ids_path])])
        if len(ids.split()) != 128:
            raise RuntimeError(f"Expected 128 output IDs: {tag}")
        (args.output / (tag + ".ids")).write_bytes(ids)
        if args.profile:
            for variable, suffix in (("trace", ".route.txt"),
                                     ("route_log", ".decode-route.txt")):
                trace = subprocess.check_output(
                    adb + ["shell", shlex.join(["cat", run_config["moe"][variable]])])
                (args.output / (tag + suffix)).write_bytes(trace)
        record = dict(tag=tag, variant=variant, prefill_ms=int(prefill[2]),
                      decode_ms=int(decode[2]), e2e_ms=int(e2e[1]),
                      inference_ms=int(inference[1]),
                      process_wall_ms=elapsed * 1000, thermal_zone0=thermal,
                      output_ids_sha256=hashlib.sha256(ids).hexdigest())
        records.append(record)
        (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    summary = {}
    for variant in dict.fromkeys(variants):
        rows = [r for r in records if r["variant"] == variant]
        summary[variant] = {"runs": len(rows)}
        for key in ("prefill_ms", "decode_ms", "inference_ms", "e2e_ms"):
            summary[variant][key] = statistics.median(r[key] for r in rows)
        summary[variant]["prefill_tps"] = 1024000 / summary[variant]["prefill_ms"]
        summary[variant]["decode_tps"] = 127000 / summary[variant]["decode_ms"]
    if "off" in summary:
        for variant in summary:
            if variant != "off":
                summary[variant]["time_delta_pct"] = {
                    key: 100 * (summary[variant][key] / summary["off"][key] - 1)
                    for key in ("prefill_ms", "decode_ms", "inference_ms", "e2e_ms")
                }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
