#!/usr/bin/env python3
"""Legacy Gemma4 compatibility probe; use validate_qwen3_cpu.py for KV development.

Greedy semantic smoke tests do not establish equivalence to a HF reference.
The complete stdout/stderr and effective configs are kept for manual review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PROMPTS = {
    "arithmetic": "What is 2 + 2? Answer with only the number.",
    "korean": "대한민국의 수도는 어디인가요? 도시 이름만 답하세요.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-kv-cpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--prompt", choices=PROMPTS, action="append")
    parser.add_argument("--skip-prefill", action="store_true",
                        help="Enable the old pack's optional prefill optimization")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    models = args.model or [Path.home() / "TEST_MODELS" / name for name in
                            ("gemma4_it_x86", "gemma4_it_x86_fp32")]
    args.output.mkdir(parents=True, exist_ok=True)
    binary = (args.build / "Applications/CausalLM/nntr_causallm").resolve()
    # Remove inherited tuning flags, including presence-tested flags set to 0.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("NNTR_") and not k.startswith("LD_")}
    env.update(NNTR_ENGINE="cpu", NNTR_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(args.threads),
               NNTR_TOKENIZER_CACHE="0",
               NNTR_LOG_DIR=str((args.output / "logs").resolve()),
               XDG_CACHE_HOME=str((args.output / "cache").resolve()))
    report = {"source_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {k: v for k, v in env.items() if k.startswith(
            ("NNTR_", "OPENBLAS_", "OMP_"))}, "runs": []}
    failed = False
    for model in models:
        model = model.expanduser().resolve()
        for prompt_name in args.prompt or PROMPTS:
            stage = (args.output / "models" / model.name / prompt_name).resolve()
            stage.mkdir(parents=True, exist_ok=True)
            for asset in model.iterdir():
                if asset.name in ("nntr_config.json", "generation_config.json"):
                    continue
                link = stage / asset.name
                if not link.exists():
                    link.symlink_to(asset)
            cfg = json.loads((model / "nntr_config.json").read_text())
            cfg.update(init_seq_len=64, max_seq_len=256,
                       num_to_generate=args.tokens, batch_size=1,
                       skip_prefill=args.skip_prefill)
            cfg["chat_input"] = {"messages": [
                {"role": "user", "content": PROMPTS[prompt_name]}]}
            (stage / "nntr_config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
            gen = json.loads((model / "generation_config.json").read_text())
            gen["do_sample"] = False
            (stage / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
            if args.prepare_only:
                continue
            print(f"Running {model.name}/{prompt_name}", flush=True)
            begin = time.monotonic()
            timed_out = False
            try:
                result = subprocess.run([str(binary), str(stage)], env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=args.timeout)
                stdout, stderr, code = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
                timed_out = True
            stem = args.output / f"{model.name}-{prompt_name}"
            stem.with_suffix(".stdout.txt").write_bytes(stdout)
            stem.with_suffix(".stderr.txt").write_bytes(stderr)
            text = stdout.decode("utf-8", errors="replace")
            generated = text.rsplit("<|turn>model\n", 1)[-1].split(
                "=================[ LLM", 1)[0].strip() if "<|turn>model\n" in text else ""
            passed = code == 0 and bool(generated)
            failed |= not passed
            run = {"model": str(model), "prompt": prompt_name, "exit_code": code,
                   "timeout": timed_out, "elapsed_s": round(time.monotonic() - begin, 3),
                   "config_sha256": hashlib.sha256((stage / "nntr_config.json").read_bytes()).hexdigest(),
                   "generated_text": generated, "completed": passed,
                   "semantic_review": "manual review required"}
            report["runs"].append(run)
            print(json.dumps(run, ensure_ascii=False), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
