#!/usr/bin/env python3
"""Compare Qwen3 CPU tokens and logits with the pinned Hugging Face checkpoint.

Run with a Python environment containing torch, transformers and numpy. The
probe records logits without modifying them. Non-thinking greedy decoding is a
reproducibility condition, not a claim of optimal model quality or performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROMPTS = {
    "arithmetic_strict": "What is 2 + 2? Answer with only the number.",
    "arithmetic": "Compute 2 + 2.",
    "korean": "대한민국의 수도는 어디인가요? 도시 이름만 답하세요.",
    "list": "List the numbers 1 to 5, separated by commas.",
    "long": "The city Paris is in France. " * 20 +
            "Which country is Paris in? Answer with just the country.",
}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-kv-cpu"))
    parser.add_argument("--hf", type=Path,
                        default=Path("build-kv-cpu/models/hf/qwen3-0.6b"))
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library-path", type=Path,
                        help="Optional preserved baseline shared libraries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt", choices=PROMPTS, action="append")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--include-session", action="store_true",
                        help="Also validate two turns using the same live KV allocation")
    parser.add_argument("--session-only", action="store_true",
                        help="Run only the two-turn session case")
    parser.add_argument("--snapshot-resume", action="store_true",
                        help="Compare live continuation with a new model restored from KV file")
    parser.add_argument("--prefill-chunk", type=int,
                        help="Explicit NNTR_PREFILL_CHUNK for multi-token step testing")
    parser.add_argument("--force-flash", action="store_true",
                        help="Set NNTR_MIN_PREFILL=1 to exercise the x86 flash consumer")
    parser.add_argument("--min-prefill", type=int,
                        help="Explicit NNTR_MIN_PREFILL threshold for consumer selection")
    parser.add_argument("--key-block", type=int,
                        help="Explicit VJEPA_BK key tile size for seam regression")
    parser.add_argument("--query-block", type=int,
                        help="Explicit VJEPA_BQ query tile size for seam regression")
    parser.add_argument("--trace", action="store_true",
                        help="Enable NNTR_KV_CPU_TRACE=1 in the nntrainer probe")
    parser.add_argument("--long-prompt", action="store_true",
                        help="Add a >128-token prompt with a 256-row activation plane")
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 for the prefill-plus-decode budget")
    if args.threads < 1 or any(value is not None and value < 1 for value in
                              (args.prefill_chunk, args.key_block, args.query_block)):
        parser.error("Threads, prefill chunk and tile sizes must be positive")
    if args.min_prefill is not None and args.min_prefill < 0:
        parser.error("--min-prefill must be nonnegative")
    if args.session_only or args.snapshot_resume:
        args.include_session = True
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    models = args.model or [args.build / "models" / name for name in
                            ("qwen3-0.6b-fp32", "qwen3-0.6b-q40")]
    prompts = list(args.prompt or [name for name in PROMPTS if name != "long"])
    if args.long_prompt and "long" not in prompts:
        prompts.append("long")
    if args.session_only:
        prompts = []
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("NNTR_", "LD_", "VJEPA_"))}
    env.update(NNTR_ENGINE="cpu", NNTR_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(args.threads),
               NNTR_TOKENIZER_CACHE="0",
               NNTR_LOG_DIR=str(args.output / "logs"),
               XDG_CACHE_HOME=str(args.output / "cache"))
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path.resolve())
    if args.force_flash:
        env["NNTR_MIN_PREFILL"] = "1"
    if args.min_prefill is not None:
        env["NNTR_MIN_PREFILL"] = str(args.min_prefill)
    if args.key_block is not None:
        env["VJEPA_BK"] = str(args.key_block)
    if args.query_block is not None:
        env["VJEPA_BQ"] = str(args.query_block)
    if args.prefill_chunk is not None:
        env["NNTR_PREFILL_CHUNK"] = str(args.prefill_chunk)
    if args.trace:
        env["NNTR_KV_CPU_TRACE"] = "1"
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(
        args.hf, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager").eval()
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff"])).hexdigest(),
        "checkpoint": json.loads((args.hf / "nntr_source_revision.json").read_text()),
        "torch_version": torch.__version__,
        "reference_attention": "eager FP32, native FP32 KV",
        "nntrainer_attention": "CPU FP32 activations, UINT16 half-bit KV",
        "enable_thinking": False,
        "do_sample": False,
        "threads": args.threads,
        "force_flash": args.force_flash,
        "consumer_environment": {k: env[k] for k in
                                 ("NNTR_MIN_PREFILL", "NNTR_PREFILL_CHUNK",
                                  "VJEPA_BK", "VJEPA_BQ") if k in env},
        "library_path": env.get("LD_LIBRARY_PATH"),
        "runs": [],
    }
    metadata = args.probe.with_suffix(".abi.json")
    if metadata.exists():
        manifest["probe_abi"] = json.loads(metadata.read_text())
    if args.library_path and (args.library_path / "sha256.json").exists():
        manifest["library_sha256"] = json.loads(
            (args.library_path / "sha256.json").read_text())
    failed = False
    for name in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[name]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        request = {"prompt_name": name, "rendered_prompt": rendered,
                   "input_ids": inputs.input_ids[0].tolist()}
        request_file = args.output / f"{name}-request.json"
        save_json(request_file, request)
        with torch.inference_mode():
            result = reference.generate(
                **inputs, do_sample=False, max_new_tokens=args.tokens,
                return_dict_in_generate=True, output_scores=True)
        ids = result.sequences[0, inputs.input_ids.shape[-1]:].tolist()
        scores = [row[0].float().numpy() for row in result.scores]
        for step, row in enumerate(scores):
            row.tofile(args.output / f"{name}-hf.step{step}.f32")
        hf = {"token_ids": ids,
              "generated_text": tokenizer.decode(ids, skip_special_tokens=False),
              "all_finite": all(bool(np.isfinite(row).all()) for row in scores)}
        save_json(args.output / f"{name}-hf.json", hf)
        print(f"HF {name}: {hf['generated_text']!r}", flush=True)
        for model in models:
            model = model.resolve()
            stage = args.output / "stages" / model.name / name
            stage.mkdir(parents=True, exist_ok=True)
            for path in model.iterdir():
                if path.name in ("nntr_config.json", "generation_config.json"):
                    continue
                if path.name.endswith(".ntkc"):
                    continue
                link = stage / path.name
                if not link.exists():
                    link.symlink_to(path)
            cfg = json.loads((model / "nntr_config.json").read_text())
            # With skip_prefill=false, run() emits prefill's first token before
            # its num_to_generate decode iterations. Align the total HF cap.
            cfg.update(num_to_generate=args.tokens - 1, skip_prefill=False)
            if name == "long":
                cfg["init_seq_len"] = 256
            save_json(stage / "nntr_config.json", cfg)
            generation = json.loads((model / "generation_config.json").read_text())
            generation["do_sample"] = False
            save_json(stage / "generation_config.json", generation)
            report_file = args.output / f"{model.name}-{name}.json"
            try:
                run = subprocess.run([str(args.probe.resolve()), str(stage),
                                      str(request_file), str(report_file)], env=env,
                                     capture_output=True, timeout=args.timeout)
            except subprocess.TimeoutExpired as error:
                report_file.with_suffix(".stdout.txt").write_bytes(error.stdout or b"")
                report_file.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
                failed = True
                manifest["runs"].append({"model": model.name, "prompt": name,
                                         "exit_code": None, "timeout": True})
                save_json(args.output / "report.json", manifest)
                continue
            report_file.with_suffix(".stdout.txt").write_bytes(run.stdout)
            report_file.with_suffix(".stderr.txt").write_bytes(run.stderr)
            if run.returncode:
                failed = True
                manifest["runs"].append({"model": model.name, "prompt": name,
                                         "exit_code": run.returncode})
                save_json(args.output / "report.json", manifest)
                continue
            report = json.loads(report_file.read_text())
            actual = report["token_ids"]
            first = np.fromfile(report["steps"][0]["logits_file"], np.float32)
            delta = first - scores[0]
            all_finite = all(row["all_finite"] for row in report["steps"])
            equal = actual == ids
            # Quantization intentionally changes logits and may change tokens.
            # FP32 uses half-bit KV, so small differences from HF are expected.
            passed = hf["all_finite"] and all_finite and (equal or cfg["fc_layer_dtype"] != "FP32")
            failed |= not passed
            item = {"model": model.name, "prompt": name, "exit_code": 0,
                    "token_ids": actual,
                    "generated_text": tokenizer.decode(actual, skip_special_tokens=False),
                    "hf_text": hf["generated_text"], "hf_token_ids_equal": equal,
                    "all_logits_finite": all_finite,
                    "prefill_argmax_equal": int(first.argmax()) == int(scores[0].argmax()),
                    "prefill_logits_max_abs_error": float(np.max(np.abs(delta))),
                    "prefill_logits_rmse": float(np.sqrt(np.mean(delta * delta))),
                    "passed": passed,
                    "semantic_review": "Manual review; token equality is not a quality benchmark"}
            manifest["runs"].append(item)
            save_json(args.output / "report.json", manifest)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    if args.include_session:
        first_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with exactly one word: hello."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        first_input = tokenizer.encode(first_prompt, add_special_tokens=False)

        def generate_ids(ids):
            tensor = torch.tensor([ids])
            with torch.inference_mode():
                result = reference.generate(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    max_new_tokens=args.tokens, do_sample=False)
            return result[0, len(ids):].tolist()

        first_ids = generate_ids(first_input)
        # The final selected token is not fed back into the KV cache until the
        # next call. Prefix the continuation with that token before turn two.
        suffix = tokenizer.decode(first_ids[-1:], skip_special_tokens=False) + "\n" + \
            tokenizer.apply_chat_template(
                [{"role": "user", "content":
                  "What word did you just say? Answer with that word only."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        second_ids = generate_ids(first_input + first_ids[:-1] + suffix_ids)
        request = {"rendered_prompt": first_prompt, "input_ids": first_input,
                   "continuations": [{"rendered_prompt": suffix, "input_ids": suffix_ids}]}
        if args.snapshot_resume:
            request["snapshot_resume"] = True
        request_file = args.output / "session-request.json"
        save_json(request_file, request)
        expected = {"first_ids": first_ids, "second_ids": second_ids,
                    "position_at_continuation": len(first_input) + len(first_ids) - 1}
        save_json(args.output / "session-hf.json", expected)
        manifest["sessions"] = []
        for model in models:
            model = model.resolve()
            report_file = args.output / f"{model.name}-session.json"
            run = subprocess.run([str(args.probe.resolve()), str(model),
                                  str(request_file), str(report_file)], env=env,
                                 capture_output=True, timeout=args.timeout)
            report_file.with_suffix(".stdout.txt").write_bytes(run.stdout)
            report_file.with_suffix(".stderr.txt").write_bytes(run.stderr)
            item = {"model": model.name, "exit_code": run.returncode, "passed": False}
            if run.returncode == 0:
                result = json.loads(report_file.read_text())
                next_turn = result["continuations"][0]
                finite = all(s["all_finite"] for s in result["steps"] + next_turn["steps"])
                equal = result["token_ids"] == first_ids and next_turn["token_ids"] == second_ids
                item.update(hf_token_ids_equal=equal, all_logits_finite=finite,
                            positions=[result["position_before"], result["position_after"],
                                       next_turn["position_before"], next_turn["position_after"]],
                            passed=equal and finite)
                if args.snapshot_resume:
                    from compare_qwen_cpu_runs import compare_turn
                    resumed = result["resumed_continuations"][0]
                    comparison = compare_turn(next_turn, resumed)
                    snapshot = result["snapshot"]
                    comparison["positions_equal"] = (
                        snapshot["saved_position"] == snapshot["restored_position"] ==
                        next_turn["position_before"] == resumed["position_before"] and
                        next_turn["position_after"] == resumed["position_after"])
                    comparison["passed"] &= comparison["positions_equal"]
                    comparison["snapshot_bytes"] = snapshot["bytes"]
                    item["snapshot_resume"] = comparison
                    item["passed"] &= comparison["passed"]
            manifest["sessions"].append(item)
            failed |= not item["passed"]
            print(json.dumps(item, ensure_ascii=False), flush=True)
            save_json(args.output / "report.json", manifest)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
