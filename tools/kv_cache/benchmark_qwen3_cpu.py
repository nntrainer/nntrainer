#!/usr/bin/env python3
"""Fixed-work Qwen CPU benchmarks on Linux or an explicitly selected Android.

No HF reference, logits dump or cache trace runs in the measured process.
Each process loads once, warms up, then measures repeated independent prompts.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(report):
    trials = [row for row in report["trials"] if not row["warmup"]]
    result = {"measured_repetitions": len(trials)}
    for key in ("run_wall_ms", "run_to_first_logits_ms", "prefill_compute_ms",
                "decode_api_ms", "decode_first_step_ms", "decode_steady_median_ms"):
        values = [row[key] for row in trials]
        result[key] = {"median": statistics.median(values), "min": min(values),
                       "max": max(values), "mean": statistics.mean(values),
                       "stdev": statistics.stdev(values) if len(values) > 1 else None}
    result["decode_logits_interval_ms"] = {
        "per_trial_medians": [statistics.median(row["decode_logits_intervals_ms"]) for row in trials]}
    result["decode_tokens_per_second"] = 1000.0 / statistics.median(
        result["decode_logits_interval_ms"]["per_trial_medians"])
    steady = result["decode_steady_median_ms"]["median"]
    prefill = result["prefill_compute_ms"]["median"]
    result["decode_steady_tokens_per_second"] = 1000.0 / steady if steady > 0 else None
    result["prefill_tokens_per_second"] = (
        report["request"]["context_tokens"] * 1000.0 / prefill if prefill > 0 else None)
    return result


def host_sensors():
    readings = {}
    patterns = ("/sys/class/thermal/thermal_zone*/temp",
                "/sys/class/thermal/thermal_zone*/type",
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq",
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    import glob
    for pattern in patterns:
        for name in glob.glob(pattern):
            try:
                readings[name] = Path(name).read_text(encoding="utf-8").strip()
            except OSError as error:
                readings[name] = str(error)
    return {"unix_time": time.time(), "sysfs": readings}


def audit_prefill_trace(text, context, chunk, owners=28):
    """Check the actual MHA intervals in one untimed prefill plus decode step."""
    rows = re.findall(r"\[kv-cpu-access\] from=(\d+) tokens=(\d+)\b", text)
    observed = Counter((int(start), int(count)) for start, count in rows)
    block = chunk or context
    expected = Counter({(start, min(block, context - start)): owners
                        for start in range(0, context, block)})
    expected[(context, 1)] = owners
    if observed != expected:
        raise ValueError(f"Actual MHA intervals differ: {dict(observed)} != {dict(expected)}")
    return {"passed": True, "owner_layers_per_step": owners,
            "prefill_graph_calls": math.ceil(context / block),
            "decode_graph_calls": 1,
            "intervals": [{"from": start, "tokens": count, "owner_calls": calls}
                          for (start, count), calls in sorted(observed.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("linux", "android"), required=True)
    parser.add_argument("--serial", help="Required for Android; no implicit device selection")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path,
                        help="Flat shared-library directory; required on Android")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("legacy-packed", "legacy-tiled", "fp32-packed", "fp32-tiled"),
                        required=True)
    parser.add_argument("--label", required=True, help="Artifact label, e.g. baseline or current")
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plane", type=int, default=128)
    prefill = parser.add_mutually_exclusive_group()
    prefill.add_argument("--prefill-chunk", type=int,
                         help="Chunk rows (default 128); must fit --plane")
    prefill.add_argument("--single-block", action="store_true",
                         help="Unset NNTR_PREFILL_CHUNK; --plane must fit the complete context")
    parser.add_argument("--verify-prefill", action="store_true",
                        help="Run a separate untimed trace to verify actual MHA intervals")
    parser.add_argument("--min-prefill", type=int, default=1)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cooldown-seconds", type=float, default=0)
    parser.add_argument("--reuse-device-models", action="store_true",
                        help="Read verified existing nntr_kv_cpu_<serial>/models without modifying them")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline-report", type=Path,
                        help="Completed benchmark output directory for workload-checked comparisons")
    parser.add_argument("--max-regression-percent", type=float, default=5.0)
    args = parser.parse_args()
    args.prefill_chunk = 0 if args.single_block else (
        128 if args.prefill_chunk is None else args.prefill_chunk)
    if args.target == "android" and (not args.serial or not args.runtime_dir):
        parser.error("Android requires --serial and --runtime-dir")
    if args.serial and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.serial):
        parser.error("Invalid device serial")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("Invalid artifact label")
    contexts = args.context or [128]
    if len(set(contexts)) != len(contexts) or len({p.name for p in args.model}) != len(args.model):
        parser.error("Context lengths and model directory names must be unique")
    if min(contexts + [args.decode_tokens, args.repetitions, args.threads,
                       args.plane]) < 1 or args.warmup < 0:
        parser.error("Context, decode, repeat, thread and plane values must be positive")
    if not args.single_block and args.prefill_chunk < 1:
        parser.error("Chunk must be positive; use --single-block to disable chunking")
    if args.single_block and max(contexts) > args.plane:
        parser.error("Single-block requires --plane at least as large as every context")
    if args.prefill_chunk > args.plane or args.min_prefill < 0:
        parser.error("Chunk must fit the activation plane; minimum prefill must be nonnegative")
    if not 0 <= args.cooldown_seconds <= 60:
        parser.error("Cooldown must be between 0 and 60 seconds")
    if args.max_regression_percent < 0:
        parser.error("Regression budget must be nonnegative")
    args.output = args.output.resolve()
    if (args.output / "report.json").exists():
        parser.error("Output already contains a report; choose a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    clean_env = {key: value for key, value in os.environ.items()
                 if not key.startswith(("NNTR_", "LD_", "VJEPA_"))}
    env_values = {"NNTR_ENGINE": "cpu", "NNTR_NUM_THREADS": str(args.threads),
                  "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": str(args.threads),
                  "NNTR_INIT_SEQ_LEN": str(args.plane),
                  "NNTR_MIN_PREFILL": str(args.min_prefill),
                  "NNTR_CPU_ATTENTION_FP32": "1" if args.profile.startswith("fp32") else "0",
                  "NNTR_CPU_KV_TILED": "1" if args.profile.endswith("tiled") else "0",
                  "NNTR_TOKENIZER_CACHE": "0"}
    if not args.single_block:
        env_values["NNTR_PREFILL_CHUNK"] = str(args.prefill_chunk)
    env_values["NNTR_KV_CPU_TRACE"] = "0"
    manifest = {"target": args.target, "serial": args.serial, "label": args.label,
                "profile": args.profile, "environment": env_values,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
                "benchmark_source_sha256": sha256(Path(__file__).with_name("qwen_cpu_benchmark.cpp")),
                "harness_sha256": sha256(Path(__file__)), "probe_sha256": sha256(args.probe),
                "warmup": args.warmup, "repetitions": args.repetitions,
                "prefill_mode": "single-block" if args.single_block else "chunked",
                "requested_plane": args.plane, "requested_chunk": args.prefill_chunk,
                "verify_prefill": args.verify_prefill,
                "cooldown_seconds": args.cooldown_seconds, "runs": []}
    manifest["host"] = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    if Path("/proc/cpuinfo").exists():
        manifest["host"]["cpu_model"] = next((line.split(":", 1)[1].strip()
                                               for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
                                               if line.startswith("model name")), None)
    baseline = (json.loads((args.baseline_report / "report.json").read_text())
                if args.baseline_report else None)
    if baseline and not baseline.get("passed"):
        parser.error("Baseline benchmark must be complete and successful")
    if baseline and (baseline["target"] != args.target or baseline["serial"] != args.serial):
        parser.error("Baseline must use the same target and explicit device")
    abi = args.probe.with_suffix(".abi.json")
    if abi.exists():
        manifest["probe_abi"] = json.loads(abi.read_text())
    if args.runtime_dir:
        manifest["runtime_sha256"] = {p.name: sha256(p) for p in args.runtime_dir.iterdir()
                                     if p.is_file() and ".so" in p.name}

    def adb(*arguments, check=True):
        return subprocess.run(["adb", "-s", args.serial, *map(str, arguments)],
                              check=check, capture_output=True, timeout=args.timeout, env=clean_env)

    def shell(*arguments, check=True):
        return adb("shell", shlex.join(map(str, arguments)), check=check)

    def device_sensors():
        result = shell("sh", "-c", "dumpsys thermalservice; dumpsys battery; "
                       "for z in /sys/class/thermal/thermal_zone*; do "
                       "echo \"$z\"; cat \"$z/type\" \"$z/temp\"; done; "
                       "for c in /sys/devices/system/cpu/cpu*/cpufreq; do "
                       "echo \"$c\"; cat \"$c/scaling_cur_freq\" \"$c/scaling_governor\"; done",
                       check=False)
        return {"unix_time": time.time(), "exit_code": result.returncode,
                "stdout": result.stdout.decode(errors="replace"),
                "stderr": result.stderr.decode(errors="replace")}

    remote = None
    if args.target == "android":
        base = PurePosixPath(f"/data/local/tmp/nntr_kv_cpu_{args.serial}")
        remote = base / "benchmarks" / f"{args.label}-{args.profile}-{args.output.name}"
        remote_models = base / ("models" if args.reuse_device_models else "benchmarks/models")
        if not args.prepare_only:
            if adb("get-state").stdout.strip() != b"device":
                raise RuntimeError("Selected device is not authorized")
            manifest["device"] = {name: shell("getprop", name).stdout.decode().strip() for name in
                                  ("ro.product.model", "ro.soc.model", "ro.build.version.release",
                                   "ro.build.version.sdk", "ro.product.cpu.abi")}
            shell("mkdir", "-p", remote / "runtime", remote / "logs")
            for path in args.runtime_dir.iterdir():
                if path.is_file() and ".so" in path.name:
                    adb("push", path, remote / "runtime" / path.name)
            adb("push", args.probe, remote / "runtime/qwen_cpu_benchmark")
            shell("chmod", "+x", remote / "runtime/qwen_cpu_benchmark")
    for model_arg in args.model:
        model = model_arg.resolve()
        config = json.loads((model / "config.json").read_text())
        original = json.loads((model / "nntr_config.json").read_text())
        weight = model / original["model_file_name"]
        quantized = original["fc_layer_dtype"] != "FP32"
        if quantized and ((args.target == "android") != ("arm" in weight.name.lower())):
            parser.error("Quantized model packing does not match the requested architecture")
        model_hash = sha256(weight)
        if args.target == "android" and not args.prepare_only:
            model_remote = remote_models / model.name
            if not args.reuse_device_models:
                shell("mkdir", "-p", model_remote)
                for name in (weight.name, "tokenizer.json"):
                    adb("push", model / name, model_remote / name)
            remote_hash = shell("sha256sum", model_remote / weight.name).stdout.decode().split()[0]
            if remote_hash != model_hash:
                raise RuntimeError("Device weight hash mismatch")
        for context in contexts:
            capacity = args.capacity or math.ceil((context + args.decode_tokens + 64) / 128) * 128
            if (capacity <= context + args.decode_tokens or capacity < args.plane or
                    capacity > config["max_position_embeddings"]):
                parser.error("Capacity must leave decode room and fit max_position_embeddings")
            name = f"{model.name}-c{context}"
            stage = args.output / "stages" / name
            stage.mkdir(parents=True, exist_ok=True)
            cfg = dict(original, max_seq_len=capacity, init_seq_len=args.plane,
                       num_to_generate=args.decode_tokens, skip_prefill=False, batch_size=1)
            if args.target == "linux":
                cfg.update(model_file_name=str(weight), tokenizer_file=str(model / "tokenizer.json"))
            else:
                cfg.update(model_file_name=str(remote_models / model.name / weight.name),
                           tokenizer_file=str(remote_models / model.name / "tokenizer.json"))
            save(stage / "nntr_config.json", cfg)
            shutil.copy2(model / "config.json", stage / "config.json")
            generation = json.loads((model / "generation_config.json").read_text())
            generation["do_sample"] = False
            save(stage / "generation_config.json", generation)
            request = args.output / f"{name}-request.json"
            save(request, {"context_tokens": context, "decode_tokens": args.decode_tokens,
                           "warmup": args.warmup, "repetitions": args.repetitions,
                           "expected_plane": args.plane,
                           "expected_prefill_chunk": args.prefill_chunk,
                           "single_block": args.single_block})
            report_path = args.output / f"{name}.json"
            item = {"model": model.name, "context": context, "capacity": capacity,
                    "weight_sha256": model_hash, "weight_bytes": weight.stat().st_size,
                    "request_sha256": sha256(request), "stage_config_sha256": sha256(stage / "nntr_config.json")}
            if args.prepare_only:
                item["prepared_only"] = True
                manifest["runs"].append(item)
                continue
            sensors = device_sensors if args.target == "android" else host_sensors
            env = dict(clean_env, **env_values)
            if args.target == "linux":
                if args.runtime_dir:
                    env["LD_LIBRARY_PATH"] = str(args.runtime_dir.resolve())
                env.update(NNTR_LOG_DIR=str(args.output / "logs"),
                           XDG_CACHE_HOME=str(args.output / "cache"))
                command = [str(args.probe.resolve()), str(stage), str(request), str(report_path)]
            else:
                remote_stage = remote / "stages" / name
                shell("mkdir", "-p", remote_stage)
                for path in stage.iterdir():
                    adb("push", path, remote_stage / path.name)
                adb("push", request, remote / request.name)
                remote_env = dict(env_values, LD_LIBRARY_PATH=str(remote / "runtime"),
                                  NNTR_LOG_DIR=str(remote / "logs"), XDG_CACHE_HOME=str(remote / "cache"))
                command = ["adb", "-s", args.serial, "shell", shlex.join([
                    "timeout", str(max(1, args.timeout - 10)), "env", "-u", "NNTR_PREFILL_CHUNK",
                    *[f"{key}={value}" for key, value in remote_env.items()],
                    str(remote / "runtime/qwen_cpu_benchmark"), str(remote_stage),
                    str(remote / request.name), str(remote / report_path.name)])]
            if args.verify_prefill:
                save(args.output / f"{name}-sensors-before-audit.json", sensors())
                audit_request = request.with_name(name + "-audit-request.json")
                audit_report = report_path.with_name(name + "-audit.json")
                audit_data = json.loads(request.read_text())
                audit_data.update(warmup=0, repetitions=1, decode_tokens=1)
                save(audit_request, audit_data)
                audit_env = dict(env, NNTR_KV_CPU_TRACE="1")
                if args.target == "linux":
                    audit_command = [str(args.probe.resolve()), str(stage),
                                     str(audit_request), str(audit_report)]
                else:
                    adb("push", audit_request, remote / audit_request.name)
                    audit_remote_env = dict(remote_env, NNTR_KV_CPU_TRACE="1")
                    audit_command = ["adb", "-s", args.serial, "shell", shlex.join([
                        "timeout", str(max(1, args.timeout - 10)), "env", "-u", "NNTR_PREFILL_CHUNK",
                        *[f"{key}={value}" for key, value in audit_remote_env.items()],
                        str(remote / "runtime/qwen_cpu_benchmark"), str(remote_stage),
                        str(remote / audit_request.name), str(remote / audit_report.name)])]
                print(f"Verifying untimed prefill: {name}", flush=True)
                try:
                    audit_run = subprocess.run(audit_command, capture_output=True,
                                               timeout=args.timeout, env=audit_env, check=False)
                    audit_exit = audit_run.returncode
                except subprocess.TimeoutExpired as error:
                    audit_run = error
                    audit_exit = None
                    item["timeout"] = True
                audit_report.with_suffix(".stdout.txt").write_bytes(audit_run.stdout or b"")
                audit_report.with_suffix(".stderr.txt").write_bytes(audit_run.stderr or b"")
                if audit_exit != 0:
                    item.update(exit_code=audit_exit,
                                prefill_audit={"passed": False, "report": str(audit_report)})
                    manifest["runs"].append(item)
                    manifest["passed"] = False
                    save(args.output / "report.json", manifest)
                    print(f"Untimed prefill audit failed; see {audit_report}", flush=True)
                    return 1
                if args.target == "android":
                    adb("pull", remote / audit_report.name, audit_report)
                audit_result = json.loads(audit_report.read_text())
                if (audit_result.get("effective_prefill_chunk") != args.prefill_chunk or
                        audit_result.get("activation_plane") != args.plane):
                    raise ValueError("Observer did not confirm the requested plane/chunk")
                item["prefill_audit"] = audit_prefill_trace(
                    audit_run.stderr.decode(errors="replace"), context, args.prefill_chunk)
                item["prefill_audit"].update(report=str(audit_report),
                                              trace_sha256=hashlib.sha256(audit_run.stderr).hexdigest())
                save(args.output / f"{name}-sensors-after-audit.json", sensors())
            # The untimed audit can heat the device: cool down after it, before timing.
            if args.cooldown_seconds:
                time.sleep(args.cooldown_seconds)
            save(args.output / f"{name}-sensors-before.json", sensors())
            print(f"Running {args.label}/{args.profile}: {name}", flush=True)
            try:
                run = subprocess.run(command, capture_output=True, timeout=args.timeout, env=env, check=False)
                item["exit_code"] = run.returncode
            except subprocess.TimeoutExpired as error:
                run = error
                item.update(exit_code=None, timeout=True)
            report_path.with_suffix(".stdout.txt").write_bytes(run.stdout or b"")
            report_path.with_suffix(".stderr.txt").write_bytes(run.stderr or b"")
            save(args.output / f"{name}-sensors-after.json", sensors())
            if item["exit_code"] == 0:
                if args.target == "android":
                    adb("pull", remote / report_path.name, report_path)
                result = json.loads(report_path.read_text())
                if args.single_block or args.verify_prefill:
                    if (result.get("effective_prefill_chunk") != args.prefill_chunk or
                            result.get("activation_plane") != args.plane):
                        raise ValueError("Timed observer used a different prefill strategy")
                item["summary"] = summarize(result)
                if baseline:
                    prior = next((row for row in baseline["runs"] if
                                  (row["model"], row["context"]) == (model.name, context)), None)
                    if prior is None:
                        raise ValueError("Baseline lacks the requested model/context")
                    prior_result = json.loads((args.baseline_report / report_path.name).read_text())
                    equal_work = (prior["weight_sha256"] == model_hash and
                                  prior["capacity"] == capacity and
                                  prior_result["input_ids"] == result["input_ids"] and
                                  prior_result["trials"][0]["forced_ids"] == result["trials"][0]["forced_ids"] and
                                  all(baseline["environment"].get(key) == env_values.get(key) for key in
                                      ("NNTR_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                       "NNTR_INIT_SEQ_LEN", "NNTR_PREFILL_CHUNK", "NNTR_MIN_PREFILL")))
                    if not equal_work:
                        raise ValueError("Baseline workload, weight, geometry or thread settings differ")
                    ratios = {key: (100.0 * (item["summary"][key]["median"] /
                                            prior["summary"][key]["median"] - 1.0)
                                    if prior["summary"][key]["median"] > 0 else None)
                              for key in ("run_wall_ms", "prefill_compute_ms", "decode_steady_median_ms")}
                    item["baseline_comparison"] = {
                        "directory": str(args.baseline_report.resolve()), "same_workload": True,
                        "median_change_percent": ratios,
                        "budget_percent": args.max_regression_percent,
                        "same_profile": baseline["profile"] == args.profile,
                        "same_arithmetic": (baseline["profile"].split("-")[0] ==
                                            args.profile.split("-")[0]),
                        "gate_applies": (baseline["profile"].split("-")[0] == args.profile.split("-")[0] and
                                         baseline["warmup"] == args.warmup and
                                         baseline["repetitions"] == args.repetitions),
                        "within_budget": all(value is not None and value <= args.max_regression_percent
                                             for value in ratios.values())}
                print(json.dumps(item["summary"], ensure_ascii=False), flush=True)
            manifest["runs"].append(item)
            save(args.output / "report.json", manifest)
    manifest["passed"] = args.prepare_only or all(row.get("exit_code") == 0 for row in manifest["runs"])
    if baseline and not args.prepare_only:
        gates = [row["baseline_comparison"]["within_budget"] for row in manifest["runs"]
                 if row.get("baseline_comparison", {}).get("gate_applies")]
        manifest["regression_gate_passed"] = all(gates) if gates else None
    save(args.output / "report.json", manifest)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
