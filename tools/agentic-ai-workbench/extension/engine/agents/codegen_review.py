# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file codegen_review.py
# @brief Review actions for generated CausalLM code (the Review tab's buttons).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Review actions for generated CausalLM code, run by the extension for each
button in the Review tab:

  python -m agents.codegen_review <review.json> <action> ...

  show
  approve    <part> [--actor A] [--comment C]
  reject     <part> [--actor A] [--comment C]
  regenerate <part> --comment C
  recheck          # after editing the files
  compile          # once nothing waits for review
  fixcompile       # send the compiler errors to Claude
  verify           # build + run vs a tiny Hugging Face fixture
  fixverify        # send a failed verification to Claude
  run [--prompt P] [--tokens N] [--repeat R] [--cpus C] [--memory M]
                   # build in Docker, load on this PC, run with the real weights, profile
  fetchweights     # download the model's weights for `run`

Each prints JSON events (the updated review, logs, errors) like the pipeline.
Approving moves a generated part to `approved` in the code bank and saves its
learned checklist box; it is reused for other models only once verified
(`verify` passing marks the approved entries verified in the bank).
"""
import argparse
import getpass
import os
import sys

from knowledge import codegen as cg
from knowledge import preflight
from knowledge import run_model as rm
from knowledge import verify
from knowledge.code_bank import CodeBank, CodeBankError

from .build_setup import emit_preflight
from .causallm_codegen import make_llm, publish
from .events import bus


# pylint: disable-next=too-many-branches,too-many-statements
def main(argv=None) -> int:
    """Command line of the Review tab's actions on a generated model's review.json."""
    parser = argparse.ArgumentParser(prog="python -m agents.codegen_review")
    parser.add_argument("review")
    parser.add_argument(
        "action",
        choices=[
            "show",
            "approve",
            "reject",
            "regenerate",
            "recheck",
            "compile",
            "fixcompile",
            "verify",
            "fixverify",
            "run",
            "fetchweights",
        ],
    )
    parser.add_argument("part", nargs="?", default="")
    parser.add_argument("--comment", default="")
    parser.add_argument("--actor", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--tokens", type=int, default=int(os.environ.get("WORKBENCH_RUN_TOKENS") or 64)
    )
    parser.add_argument(
        "--repeat", type=int, default=int(os.environ.get("WORKBENCH_RUN_REPEATS") or 3)
    )
    parser.add_argument("--cpus", default=os.environ.get("WORKBENCH_RUN_CPUS") or "")
    parser.add_argument("--memory", default=os.environ.get("WORKBENCH_RUN_MEMORY") or "")
    parser.add_argument("--target", default="pc")
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="run: then one more run for the time of every layer (builds a profiling copy of "
        "nntrainer the first time)",
    )
    args = parser.parse_args(argv)

    actor = args.actor or _user()
    try:
        bank = CodeBank()
        if args.action == "show":
            review = cg.load_review(args.review)
        elif args.action in ("approve", "reject"):
            if not args.part:
                raise cg.CodegenError(f"{args.action} needs a part id")
            review = cg.set_part_status(
                args.review, args.part, args.action, actor, bank, reason=args.comment
            )
            bus.log(f"{args.part}: {args.action}d by {actor}")
        elif args.action == "recheck":
            review = cg.recheck(args.review, bank)
            validation = review["validation"]
            bus.log(
                (
                    "Re-check: every check passed"
                    if validation["ok"]
                    else f"Re-check: {len(validation['errors'])} problem(s)"
                ),
                "info" if validation["ok"] else "warn",
            )
            for error in validation["errors"]:
                bus.log(f"  {error}", "warn")
        elif args.action == "compile":
            bus.log(
                "Compile check: compiling the generated sources against the nntrainer headers..."
            )
            review = cg.compile_generated(args.review, log=bus.log)
            report_compile(review.get("compile") or {})
        elif args.action == "fixcompile":
            review = cg.fix_compile_errors(
                args.review,
                bank,
                make_llm(),
                max_attempts=int(os.environ.get("WORKBENCH_CODEGEN_ATTEMPTS") or 3),
                log=bus.log,
            )
        elif args.action == "verify":
            docker_ready(verify.select_executor().kind, "verify")
            bus.log("Verify: building the generated model and comparing it with Hugging Face...")
            review = verify.verify_generated(args.review, bank, log=bus.log)
            report_verify(review.get("verify") or {})
        elif args.action in ("run", "fetchweights"):
            return _run_action(args)
        elif args.action == "fixverify":
            review = cg.fix_verify_errors(
                args.review,
                bank,
                make_llm(),
                max_attempts=int(os.environ.get("WORKBENCH_CODEGEN_ATTEMPTS") or 3),
                log=bus.log,
            )
        else:
            if not args.part:
                raise cg.CodegenError("regenerate needs a part id")
            review = cg.regenerate(
                args.review,
                args.part,
                args.comment,
                bank,
                make_llm(),
                max_attempts=int(os.environ.get("WORKBENCH_CODEGEN_ATTEMPTS") or 3),
                log=bus.log,
            )
        publish(review)
        return 0
    except (cg.CodegenError, CodeBankError, OSError, ValueError) as exc:
        bus.error("codegen_review", str(exc))
        return 1


def _run_action(args) -> int:
    """run / fetchweights: their own errors (no weights, Docker not running) are
    reported as a `run_status` event the extension can act on."""
    try:
        review = cg.load_review(args.review)
        if args.action == "fetchweights":
            weights = rm.download_weights(review, log=bus.log)
            bus.log(f"Weights ready in {weights} ({rm.weights_size(weights) / 1e9:.2f} GB).")
            bus.emit({"event": "run_status", "status": "weights_ready", "path": str(weights)})
            return 0
        docker_ready("docker", "run")
        bus.emit({"event": "run_status", "status": "running", "target": args.target})
        review = rm.run_generated(
            args.review,
            prompt=args.prompt,
            tokens=args.tokens,
            repeat=args.repeat,
            cpus=args.cpus or None,
            memory=args.memory or None,
            target=args.target,
            log=bus.log,
            per_layer=args.per_layer,
        )
    except (rm.RunError, cg.CodegenError, OSError, ValueError) as exc:
        msg = str(exc)
        status = "no_weights" if msg.startswith("NO_WEIGHTS") else "error"
        if status == "no_weights":
            msg = msg.split(":", 1)[1].strip()
        bus.error("run", msg)
        bus.emit(
            {
                "event": "run_status",
                "status": status,
                "message": msg,
                "model": (
                    (cg.load_review(args.review) or {}).get("model", "")
                    if os.path.exists(args.review)
                    else ""
                ),
            }
        )
        return 1
    rec = review.get("run") or {}
    report_run(rec)
    bus.profile(rm.profile_event(rec, review.get("model", "")))
    bus.emit(
        {
            "event": "run_status",
            "status": "done" if rec.get("ok") else "error",
            "message": rec.get("error", ""),
        }
    )
    publish(review)
    return 0 if rec.get("ok") else 1


def docker_ready(host_kind: str, step: str) -> None:
    """Before a Docker step: the system check (knowledge/preflight.py). A
    failure this step needs raises CodegenError with the summary; the
    details and fixes are logged and sent to the Review tab."""
    if host_kind != "docker":
        return
    result = preflight.gate(step, bus.log, emit_preflight)
    if not result["ok"]:
        raise cg.CodegenError(result["summary"])


def report_run(res: dict) -> None:
    """Report a Build, Run & Profile result in the log and the webview."""
    summary = res.get("summary") or {}
    if res.get("ok"):
        bus.log(
            f"Ran on {res.get('device')}: prefill {summary.get('prefill_tps')} tok/s, generation"
            f" {summary.get('generation_tps')} tok/s, peak memory"
            f" {round((res.get('peak_memory_kb') or 0) / 1024)} MB (average of"
            f" {len(res.get('runs') or [])} run(s))."
        )
        if res.get("text"):
            bus.log("Output: " + res["text"][:500])
    else:
        bus.log("Run failed: " + str(res.get("error") or "see the log"), "error")


def report_compile(compile_res: dict) -> None:
    """Log a compile-check result (also used by the pipeline's compiler step)."""
    status = compile_res.get("status")
    if status == "ok":
        bus.log(
            f"Compile check passed ({os.path.basename(compile_res.get('compiler', ''))}, flags from"
            f" {compile_res.get('mode')})"
        )
    elif status == "failed":
        bus.log(f"Compile check failed: {len(compile_res.get('errors', []))} error(s)", "error")
        for error in compile_res.get("errors", [])[:12]:
            bus.log(f"  {error['file']}:{error['line']}: {error['message']}", "error")
    else:
        bus.log(compile_res.get("log", "Compile check did not run"), "warn")


def report_verify(verify_res: dict) -> None:
    """Log a verify result (also used by the pipeline's compiler step)."""
    status = verify_res.get("status")
    if status == "passed":
        num = len(verify_res.get("marked_verified") or [])
        bank = (f"{num} code bank entr{'y' if num == 1 else 'ies'} newly verified." if num
                else "No new code bank entries: the code it uses was already verified.")
        bus.log(
            "Verified against Hugging Face: max |logit diff|"
            f" {verify_res.get('max_abs_diff')} (atol {verify_res.get('atol')}), greedy tokens"
            f" match. {bank}"
        )
    elif status == "mismatch":
        bus.log(
            f"Not verified: max |logit diff| {verify_res.get('max_abs_diff')} (atol"
            f" {verify_res.get('atol')}), tokens {verify_res.get('tokens_got')} vs Hugging Face"
            f" {verify_res.get('tokens_ref')}",
            "error",
        )
        for failure in (verify_res.get("failures") or [])[:6]:
            bus.log("  " + " ".join(failure.split())[:300], "error")
    elif status == "build_failed":
        bus.log(f"Verify build failed: {len(verify_res.get('errors') or [])} error(s)", "error")
        for error in (verify_res.get("errors") or [])[:12]:
            bus.log(f"  {error['file']}:{error['line']}: {error['message']}", "error")
    else:
        bus.log(f"Verify {status}: " + str(verify_res.get("log", ""))[-1500:], "warn")


def _user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):            # no login name in this environment
        return "reviewer"


if __name__ == "__main__":
    sys.exit(main())
