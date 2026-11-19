#!/usr/bin/env python3
"""Validate Qwen3 CPU execution on one explicitly selected Android device.

Models stay in ignored build directories. ARM quantized weights must be packed
with nntr_quantize --isa ARM; X86-packed weights cannot be reused on Android.
Run --prepare-only offline first, then supply the Android probe and runtime.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from compare_qwen_cpu_runs import compare_turn
from validate_qwen3_cpu import PROMPTS, save_json


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def references(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager").eval()

    def render(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def generate(ids, prefix):
        inputs = torch.tensor([ids])
        with torch.inference_mode():
            result = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=args.tokens,
                return_dict_in_generate=True, output_scores=True)
        generated = result.sequences[0, len(ids):].tolist()
        rows = []
        for index, score in enumerate(result.scores):
            row = score[0].float().numpy()
            output = args.output / f"{prefix}.step{index}.f32"
            row.tofile(output)
            rows.append({"logits_file": str(output), "vocab_size": len(row),
                         "all_finite": bool(np.isfinite(row).all())})
        return {"token_ids": generated, "steps": rows,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=False)}

    names = args.prompt or ["arithmetic", "korean", "long"]
    if args.session_only:
        names = []
    cases = []
    for name in names:
        prompt = render(PROMPTS[name])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        request = {"rendered_prompt": prompt, "input_ids": ids}
        save_json(args.output / f"{name}-request.json", request)
        reference = generate(ids, f"{name}-hf")
        save_json(args.output / f"{name}-hf.json", reference)
        cases.append(name)
        print(f"HF {name}: {reference['generated_text']!r}", flush=True)
    if args.include_session or args.session_only or args.snapshot_resume:
        prompt = render("Reply with exactly one word: hello.")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        first = generate(ids, "session-hf")
        suffix = tokenizer.decode(first["token_ids"][-1:], skip_special_tokens=False) + "\n" + render(
            "What word did you just say? Answer with that word only.")
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        second = generate(ids + first["token_ids"][:-1] + suffix_ids, "session-hf.turn1")
        request = {"rendered_prompt": prompt, "input_ids": ids,
                   "continuations": [{"rendered_prompt": suffix, "input_ids": suffix_ids}],
                   "snapshot_resume": args.snapshot_resume}
        save_json(args.output / "session-request.json", request)
        first["continuations"] = [second]
        save_json(args.output / "session-hf.json", first)
        cases.append("session")
        print(f"HF session: {first['generated_text']!r} -> {second['generated_text']!r}", flush=True)
    save_json(args.output / "reference.json", {
        "cases": cases, "tokens": args.tokens, "torch_version": torch.__version__,
        "checkpoint": json.loads((args.hf / "nntr_source_revision.json").read_text()),
        "attention": "eager FP32 with native FP32 KV", "enable_thinking": False})
    return tokenizer


def localize_report(report, directory):
    for step in report["steps"]:
        step["logits_file"] = str(directory / PurePosixPath(step["logits_file"]).name)
    for key in ("continuations", "resumed_continuations"):
        for turn in report.get(key, []):
            localize_report(turn, directory)
    return report


def recorded_files(report):
    files = [row["logits_file"] for row in report["steps"]]
    for key in ("continuations", "resumed_continuations"):
        for turn in report.get(key, []):
            files.extend(recorded_files(turn))
    if "snapshot" in report:
        files.append(report["snapshot"]["path"])
    return files


def numerical_comparison(actual, reference):
    import numpy as np

    a = np.fromfile(actual["steps"][0]["logits_file"], dtype=np.float32)
    b = np.fromfile(reference["steps"][0]["logits_file"], dtype=np.float32)
    if a.shape != b.shape or a.size != actual["steps"][0]["vocab_size"]:
        raise ValueError("Invalid or truncated first-step logits")
    delta = a - b
    return {"token_ids_equal": actual["token_ids"] == reference["token_ids"],
            "all_logits_finite": all(step["all_finite"] for step in actual["steps"]),
            "prefill_argmax_equal": int(a.argmax()) == int(b.argmax()),
            "prefill_logits_max_abs_error": float(np.max(np.abs(delta))),
            "prefill_logits_rmse": float(np.sqrt(np.mean(delta * delta)))}


def trace_observations(path):
    """Record ARM compatibility copies without misclassifying them as zero-copy."""
    text = path.read_text(errors="replace")
    headers = re.findall(r"\[kv-cpu-storage\] backing_bytes=(\d+) owner_layers=(\d+)", text)
    allocations = re.findall(
        r"\[kv-cpu-storage\] layer=(\d+) key=(0x[0-9a-f]+) value=(0x[0-9a-f]+) "
        r"plane_bytes=(\d+) source=(-?\d+)", text)
    accesses = re.findall(
        r"\[kv-cpu-access\] from=(\d+) tokens=(\d+) key=(0x[0-9a-f]+) "
        r"value=(0x[0-9a-f]+) read_alias=(\d+) write_alias=(\d+) "
        r"append_copy_bytes=(\d+) transform_bytes=(\d+)", text)
    pairs = {(row[1], row[2]) for row in allocations}
    reads = re.findall(r"\[kv-cpu-read\] consumer=([^ ]+) history_copy_bytes=(\d+)", text)
    layouts = re.findall(r"\[kv-cpu-layout\] operation=([^ ]+) history_copy_bytes=(\d+)", text)
    identity = bool(accesses) and all((row[2], row[3]) in pairs for row in accesses)
    copies = sum(int(row[1]) for row in reads + layouts)
    tile_pack = sum(map(int, re.findall(r"tile_pack_bytes=(\d+)", text)))
    tile_scratch = max(map(int, re.findall(r"tile_scratch_bytes=(\d+)", text)), default=0)
    tiled = any(row[0] == "flash-tiled" for row in reads)
    bounded_tiles = not tiled or (tile_pack > 0 and 0 < tile_scratch <= 65536)
    aliases = bool(accesses) and all(row[4:6] == ("1", "1") for row in accesses)
    # Reuse the Linux strict auditor whenever this run actually used a direct
    # consumer. ARM's flash-packed compatibility path is reported separately.
    strict = None
    if reads and copies == 0 and "profile=compatibility" not in text:
        from audit_qwen_cpu_trace import audit
        try:
            strict = audit(path)
        except ValueError as error:
            strict = {"passed": False, "error": str(error)}
    owner_counts = sum(int(owners) for _, owners in headers)
    owner_planes = [row for row in allocations if row[4] == "-1"]
    geometry = bool(headers) and all(int(owners) == 28 for _, owners in headers) and (
        len(owner_planes) == owner_counts and
        sum(int(row[3]) * 2 for row in owner_planes) == sum(int(size) for size, _ in headers))
    groups = defaultdict(Counter)
    for row in accesses:
        groups[row[0], row[1]][row[2], row[3]] += 1
    complete_steps = bool(groups) and all(
        len(group) % 28 == 0 and all(count == 1 for count in group.values())
        for group in groups.values())
    return {"storage_instances": len(headers),
            "storage_geometry": [{"backing_bytes": int(size), "owner_layers": int(owners)}
                                 for size, owners in headers],
            "owner_allocation_lines": len(allocations), "producer_calls": len(accesses),
            "allocation_identity_matches": identity, "read_write_aliases": aliases,
            "append_copy_bytes": sum(int(row[6]) for row in accesses) if accesses else None,
            "transform_bytes": sum(int(row[7]) for row in accesses) if accesses else None,
            "history_copy_bytes": copies,
            "tile_pack_bytes": tile_pack,
            "max_tile_scratch_bytes_per_worker": tile_scratch,
            "bounded_tile_scratch": bounded_tiles,
            "consumer_paths": sorted({row[0] for row in reads}),
            "compatibility_profile": "profile=compatibility" in text,
            "full_prefix_copy_free": bool(reads) and copies == 0,
            "direct_history_read_observed": bool(reads) and copies == 0 and not tiled,
            "owner_geometry_matches": geometry, "complete_owner_steps": complete_steps,
            "strict_direct_read_audit": strict,
            "scope": "Instrumented software operations; not hardware memory traffic"}


def trace_passed(observation, allow_compatibility):
    if not (observation["owner_geometry_matches"] and observation["bounded_tile_scratch"]):
        return False
    if allow_compatibility:
        return True
    strict = observation["strict_direct_read_audit"]
    return bool(strict and strict["passed"] and
                observation["append_copy_bytes"] == 0 and
                observation["history_copy_bytes"] == 0 and
                all(observation[key] for key in ("allocation_identity_matches",
                                                "read_write_aliases", "complete_owner_steps")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="Explicit target device serial")
    parser.add_argument("--remote-root", help="Dedicated root; defaults to nntr_kv_cpu_<serial>")
    parser.add_argument("--hf", type=Path, default=Path("build-kv-cpu/models/hf/qwen3-0.6b"))
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linux-reference", type=Path,
                        help="Existing CPU harness output with matching requests and model names")
    parser.add_argument("--android-baseline", type=Path,
                        help="Preserved same-device reports for token and full-logit comparison")
    parser.add_argument("--require-baseline-bitwise", action="store_true",
                        help="Fail when any turn differs from --android-baseline")
    parser.add_argument("--prompt", choices=PROMPTS, action="append")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--min-prefill", type=int)
    parser.add_argument("--include-session", action="store_true")
    parser.add_argument("--session-only", action="store_true")
    parser.add_argument("--snapshot-resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-reference", action="store_true")
    parser.add_argument("--skip-model-push", action="store_true")
    parser.add_argument("--allow-compatibility-trace", action="store_true",
                        help="Baseline mode: record missing alias/copy witnesses without failing functional parity")
    parser.add_argument("--cpu-attention-fp32", action="store_true",
                        help="Opt in to preserving FP32 query/attention inputs on CPU")
    parser.add_argument("--cpu-kv-tiled", action="store_true",
                        help="Opt in to bounded ARM K/V tile packing instead of full-prefix packing")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.serial):
        parser.error("Invalid explicit device serial")
    if args.remote_root is None:
        args.remote_root = f"/data/local/tmp/nntr_kv_cpu_{args.serial}"
    if args.remote_root != f"/data/local/tmp/nntr_kv_cpu_{args.serial}":
        parser.error("Remote root must be the dedicated /data/local/tmp/nntr_kv_cpu_<serial> directory")
    if args.tokens < 2 or args.threads < 1 or (args.prefill_chunk is not None and args.prefill_chunk < 1):
        parser.error("Invalid token, thread or chunk count")
    if args.min_prefill is not None and args.min_prefill < 0:
        parser.error("--min-prefill must be nonnegative")
    if args.require_baseline_bitwise and not args.android_baseline:
        parser.error("--require-baseline-bitwise requires --android-baseline")
    if not args.prepare_only and (not args.probe or not args.runtime_dir):
        parser.error("Device execution requires --probe and --runtime-dir")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_reference:
        # Existing requests already contain rendered prompts and exact IDs.
        # Only decoding is needed here; avoid importing torch/transformers.
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.hf / "tokenizer.json"))
        if json.loads((args.output / "reference.json").read_text())["tokens"] != args.tokens:
            parser.error("Existing reference token budget differs")
    else:
        tokenizer = references(args)
    reference = json.loads((args.output / "reference.json").read_text())
    cases = reference["cases"]
    models = args.model or [Path("build-kv-cpu/models/qwen3-0.6b-fp32"),
                            Path("build-kv-android/models/qwen3-0.6b-q40")]
    manifest = {"serial": args.serial, "remote_root": args.remote_root,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
                "reference": reference, "threads": args.threads,
                "harness_sha256": sha256(Path(__file__)),
                "allow_compatibility_trace": args.allow_compatibility_trace,
                "cpu_attention_fp32": args.cpu_attention_fp32,
                "cpu_kv_tiled": args.cpu_kv_tiled,
                "android_baseline": str(args.android_baseline) if args.android_baseline else None,
                "require_baseline_bitwise": args.require_baseline_bitwise,
                "models": [], "runs": []}
    for model in models:
        cfg = json.loads((model / "nntr_config.json").read_text())
        weight = model / cfg["model_file_name"]
        if cfg["fc_layer_dtype"] != "FP32" and "arm" not in weight.name.lower():
            parser.error("Quantized input must have an explicit ARM-packed weight filename")
        manifest["models"].append({"name": model.name, "weight": str(weight.resolve()),
                                   "bytes": weight.stat().st_size, "sha256": sha256(weight),
                                   "fc_dtype": cfg["fc_layer_dtype"]})
    save_json(args.output / "report.json", manifest)
    if args.prepare_only:
        print(f"Prepared {len(cases)} request/reference cases in {args.output}")
        return 0

    adb_env = {key: value for key, value in os.environ.items()
               if not key.startswith(("NNTR_", "LD_", "VJEPA_"))}

    def adb(*arguments, timeout=args.timeout):
        return subprocess.run(["adb", "-s", args.serial, *map(str, arguments)],
                              check=True, capture_output=True, timeout=timeout, env=adb_env)

    def shell(*arguments, timeout=args.timeout):
        return adb("shell", shlex.join(map(str, arguments)), timeout=timeout)

    if adb("get-state").stdout.strip() != b"device":
        raise RuntimeError("Selected Android device is not authorized")
    properties = shell("sh", "-c", "getprop ro.product.model; getprop ro.build.version.release; "
                       "getprop ro.build.version.sdk; getprop ro.product.cpu.abi; "
                       "getprop ro.soc.model; cat /proc/meminfo | head -n 3").stdout.decode()
    manifest["device"] = properties.splitlines()
    base = PurePosixPath(args.remote_root)
    remote_run = base / "runs" / args.output.name
    runtime = base / "runtime"
    shell("mkdir", "-p", runtime, remote_run, base / "models", base / "cache", base / "logs")
    for path in sorted(args.runtime_dir.iterdir()):
        if path.is_file() and (".so" in path.name or path.name == args.probe.name):
            adb("push", path, runtime / path.name)
    adb("push", args.probe, runtime / "qwen_cpu_probe")
    shell("chmod", "+x", runtime / "qwen_cpu_probe")
    manifest["runtime_sha256"] = {path.name: sha256(path) for path in args.runtime_dir.iterdir()
                                  if path.is_file() and ".so" in path.name}
    manifest["probe_sha256"] = sha256(args.probe)
    abi = args.probe.with_suffix(".abi.json")
    if abi.exists():
        manifest["probe_abi"] = json.loads(abi.read_text())
    for model, metadata in zip(models, manifest["models"]):
        remote_model = base / "models" / model.name
        shell("mkdir", "-p", remote_model)
        cfg = json.loads((model / "nntr_config.json").read_text())
        weight = model / cfg["model_file_name"]
        if not args.skip_model_push:
            for name in (weight.name, "config.json", "generation_config.json", "tokenizer.json"):
                adb("push", model / name, remote_model / name)
        device_hash = shell("sha256sum", remote_model / weight.name).stdout.decode().split()[0]
        if device_hash != metadata["sha256"]:
            raise RuntimeError(f"Device model SHA256 mismatch: {model.name}")
        metadata["device_sha256"] = device_hash
    env = {"NNTR_ENGINE": "cpu", "NNTR_NUM_THREADS": str(args.threads),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": str(args.threads),
           "NNTR_TOKENIZER_CACHE": "0", "NNTR_KV_CPU_TRACE": "1",
           "NNTR_CPU_ATTENTION_FP32": "1" if args.cpu_attention_fp32 else "0",
           "NNTR_CPU_KV_TILED": "1" if args.cpu_kv_tiled else "0",
           "LD_LIBRARY_PATH": str(runtime), "NNTR_LOG_DIR": str(base / "logs"),
           "XDG_CACHE_HOME": str(base / "cache")}
    if args.prefill_chunk is not None:
        env["NNTR_PREFILL_CHUNK"] = str(args.prefill_chunk)
    if args.min_prefill is not None:
        env["NNTR_MIN_PREFILL"] = str(args.min_prefill)
    manifest["environment"] = env
    failed = False
    for name in cases:
        request = args.output / f"{name}-request.json"
        adb("push", request, remote_run / request.name)
        hf = json.loads((args.output / f"{name}-hf.json").read_text())
        for model, metadata in zip(models, manifest["models"]):
            filename = f"{model.name}-{name}.json"
            remote_report = remote_run / filename
            remote_stage = remote_run / "stages" / model.name / name
            shell("mkdir", "-p", remote_stage)
            cfg = json.loads((model / "nntr_config.json").read_text())
            cfg.update(num_to_generate=args.tokens - 1, skip_prefill=False,
                       init_seq_len=256 if name == "long" else 128,
                       model_file_name=str(base / "models" / model.name / cfg["model_file_name"]),
                       tokenizer_file=str(base / "models" / model.name / "tokenizer.json"))
            stage = args.output / "stages" / model.name / name
            stage.mkdir(parents=True, exist_ok=True)
            save_json(stage / "nntr_config.json", cfg)
            for config in ("nntr_config.json", "config.json", "generation_config.json"):
                source = stage / config if config == "nntr_config.json" else model / config
                adb("push", source, remote_stage / config)
            command = ["timeout", str(max(1, args.timeout - 10)), "env",
                       *[f"{key}={value}" for key, value in env.items()],
                       str(runtime / "qwen_cpu_probe"), str(remote_stage),
                       str(remote_run / request.name), str(remote_report)]
            started = time.monotonic()
            item = {"model": model.name, "prompt": name, "passed": False}
            try:
                run = adb("shell", shlex.join(command))
                item["exit_code"] = run.returncode
            except subprocess.CalledProcessError as error:
                run = error
                item["exit_code"] = error.returncode
            except subprocess.TimeoutExpired as error:
                run = error
                item.update(exit_code=None, timeout=True)
            item["wall_seconds"] = time.monotonic() - started
            report_file = args.output / filename
            report_file.with_suffix(".stdout.txt").write_bytes(run.stdout or b"")
            stderr = report_file.with_suffix(".stderr.txt")
            stderr.write_bytes(run.stderr or b"")
            if item["exit_code"] == 0:
                pulled = args.output / "device-results"
                pulled.mkdir(exist_ok=True)
                adb("pull", remote_report, report_file)
                report = json.loads(report_file.read_text())
                for remote_file in recorded_files(report):
                    adb("pull", remote_file, pulled / PurePosixPath(remote_file).name)
                report = localize_report(report, pulled)
                save_json(report_file, report)
                comparison = numerical_comparison(report, hf)
                item.update(comparison)
                item["generated_text"] = tokenizer.decode(report["token_ids"], skip_special_tokens=False)
                item["hf_text"] = hf["generated_text"]
                item["passed"] = comparison["all_logits_finite"] and (
                    comparison["token_ids_equal"] or metadata["fc_dtype"] != "FP32")
                if args.linux_reference and (args.linux_reference / filename).exists():
                    linux = json.loads((args.linux_reference / filename).read_text())
                    item["linux_comparison"] = numerical_comparison(report, linux)
                if args.android_baseline:
                    baseline = json.loads((args.android_baseline / filename).read_text())
                    item["android_baseline"] = compare_turn(baseline, report)
                    prior_turns = baseline.get("continuations", [])
                    next_turns = report.get("continuations", [])
                    if len(prior_turns) != len(next_turns):
                        raise ValueError("Baseline continuation count differs")
                    item["android_baseline"]["continuations"] = [
                        compare_turn(prior, current) for prior, current in zip(prior_turns, next_turns)]
                    item["android_baseline"]["passed"] &= all(
                        turn["passed"] for turn in item["android_baseline"]["continuations"])
                    if args.require_baseline_bitwise:
                        item["passed"] &= item["android_baseline"]["passed"]
                if report.get("continuations"):
                    next_turn = report["continuations"][0]
                    item["second_turn"] = numerical_comparison(next_turn, hf["continuations"][0])
                    item["second_turn"]["generated_text"] = tokenizer.decode(
                        next_turn["token_ids"], skip_special_tokens=False)
                    item["positions"] = [report["position_before"], report["position_after"],
                                         next_turn["position_before"], next_turn["position_after"]]
                    item["passed"] &= item["second_turn"]["all_logits_finite"] and (
                        item["second_turn"]["token_ids_equal"] or metadata["fc_dtype"] != "FP32")
                    if report.get("resumed_continuations"):
                        restored = report["resumed_continuations"][0]
                        item["snapshot_resume"] = compare_turn(next_turn, restored)
                        snapshot = report["snapshot"]
                        positions_match = (snapshot["saved_position"] == snapshot["restored_position"] ==
                                           next_turn["position_before"] == restored["position_before"] and
                                           next_turn["position_after"] == restored["position_after"])
                        item["snapshot_resume"].update(positions_equal=positions_match,
                                                       snapshot_bytes=snapshot["bytes"])
                        item["passed"] &= item["snapshot_resume"]["passed"] and positions_match
                item["trace"] = trace_observations(stderr)
                item["trace_validation_passed"] = trace_passed(
                    item["trace"], args.allow_compatibility_trace)
                item["passed"] &= item["trace_validation_passed"]
            manifest["runs"].append(item)
            failed |= not item["passed"]
            save_json(args.output / "report.json", manifest)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    manifest["passed"] = not failed
    save_json(args.output / "report.json", manifest)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
