# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file selftest.py
# @brief Docker self-test of the x86 build path, with no Claude call or model download.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Docker self-test: proves the whole x86 Docker path on this computer with no
Claude call and no model download.

    result = run_selftest(log=print)
    python -m agents.build_setup selftest

1. the system check (knowledge/preflight.py) for setup
2. the Docker build set up (verify.setup_build, host docker)
3. the FULL build in the container: ninja -C /work/src/build-wb -j N (every
   target, like upstream's CI), with the out-of-memory retry
4. upstream's tiny Qwen3 fixture unpacked from packaging/ into the container's
   copy of test/unittest/models (never into the checkout)
5. upstream's own reference test, unittest_causallm_qwen3_reference.cpp, and
   causallm_test_utils.cpp compiled with the build's flags and linked like
   nntr_causallm + gtest (verify.compile_source / verify.link_objects)
6. its FP32 check run: nntrainer's Qwen3 against Hugging Face's logits and
   greedy tokens (fp32 atol 0.01), the rule Verify uses

The result goes to tools/agentic-ai-workbench/docker/selftest/result.json
(read by the system check) and the full log to log.txt next to it.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from . import preflight
from . import verify

Log = Callable[[str], None]
FIXTURE_TAR = "packaging/causallm_reference_generation.tar.gz"
FIXTURE_DEST = "test/unittest/models"
TEST_SRC = "test/unittest/models/unittest_causallm_qwen3_reference.cpp"
UTILS_SRC = "test/unittest/models/causallm_test_utils.cpp"
TEST_NAME = "FP32MatchesHFReference"
TEST_FILTER = "Qwen3DifferentialTest." + TEST_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def result_dir(root: Path) -> Path:
    """Where the self-test writes its result and log."""
    return Path(root) / "tools" / "agentic-ai-workbench" / "docker" / "selftest"


# pylint: disable-next=R0914,R0913,R0912,R0915,R0911
def run_selftest(
    nntrainer_root=None,
    log: Log = print,
    full_build: bool = True,
    timeout: int = 7200,
    ex=None,
    gate=None,
    setup=None,
) -> dict:
    """Run every stage; stop at the first that fails. `ex`, `gate` and `setup`
    replace the Docker executor, the system check and the build setup (tests)."""
    root = Path(verify.resolve_root(nntrainer_root)).resolve()
    out = result_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stages: List[Dict] = []
    logs: List[str] = []
    result = {
        "at": _now(),
        "ok": False,
        "status": "",
        "summary": "",
        "stages": stages,
        "test": TEST_FILTER,
    }

    def stage(name: str, is_ok: bool, start: float, detail: str = "") -> bool:
        stages.append(
            {
                "stage": name,
                "ok": is_ok,
                "seconds": round(time.time() - start, 1),
                "detail": detail[:400],
            }
        )
        log(
            f"Self-test: {name} -- {'ok' if is_ok else 'FAILED'}"
            + (f" ({detail[:200]})" if detail else "")
        )
        return is_ok

    def finish(status: str, summary: str) -> dict:
        result.update(
            status=status,
            summary=summary,
            ok=status == "passed",
            seconds=round(time.time() - started),
        )
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "log.txt").write_text("\n\n".join(logs)[-200000:], encoding="utf-8")
        log(f"Docker self-test {'passed' if result['ok'] else 'FAILED'}: {summary} "
            f"({round(result['seconds'] / 60, 1)} min)")
        return result

    # 1. system check
    start = time.time()
    if gate is None:
        check = preflight.gate("setup", lambda m, level="info": log(m))
    else:
        check = gate("setup")
    if not stage("system check", check["ok"], start, "" if check["ok"] else check["summary"]):
        return finish("preflight_failed", check["summary"])

    # 2. the Docker build set up
    start = time.time()
    state = (setup or (lambda: verify.setup_build(root, log=log, host="docker")))()
    if not stage(
        "Docker build set up",
        bool(state.get("ok")),
        start,
        state.get("why") or state.get("status", ""),
    ):
        logs.append(str(state.get("log", "")))
        return finish(
            "setup_failed",
            state.get("why") or state.get("status") or "the Docker build did not set up",
        )
    ex = ex or verify.DockerExecutor(root)
    bdir = ex.build_dir

    # 3. the full build (or only the libraries the test links)
    start = time.time()
    targets: List[str] = [] if full_build else verify.build_status(root, ex=ex).get("libs", [])
    jobs = verify.docker_jobs(ex)
    log(f"Self-test: {'full build (every target)' if full_build else 'building the libraries'} "
        f"with {jobs or 'default'} parallel jobs; the first time takes a while ...")
    code, text, jobs = verify.run_ninja(ex, bdir, targets, timeout, log, jobs=jobs)
    ex.after_build()
    logs.append(text)
    if not stage(
        "full build" if full_build else "library build",
        code == 0,
        start,
        f"{jobs} jobs" if code == 0 else verify.build_failure_why(text, code, jobs, "ninja failed"),
    ):
        return finish(
            "build_failed",
            verify.build_failure_why(text, code, jobs, "the build failed: see log.txt"),
        )

    # 4. upstream's tiny fixture, inside the container's copy
    start = time.time()
    code, text = ex.run(["tar", "-xzf", FIXTURE_TAR, "-C", FIXTURE_DEST], root, 600)
    logs.append(text)
    if not stage("fixture unpacked", code == 0, start, text.strip()[-300:] if code else ""):
        return finish("fixture_failed", "could not unpack " + FIXTURE_TAR)

    # 5. compile + link upstream's reference test
    start = time.time()
    ninja = verify.Ninja(ex.ninja_file())
    work = bdir / "wb_selftest"
    utils = root / UTILS_SRC
    utils_obj = (
        bdir
        / "wb_verify"
        / f"causallm_test_utils-{hashlib.sha256(utils.read_bytes()).hexdigest()[:10]}.o"
    )
    jobs_c = [(root / TEST_SRC, work / "unittest_causallm_qwen3_reference.o")]
    if not ex.exists(utils_obj):                               # shared with Verify's cached object
        jobs_c.append((utils, utils_obj))
    ex.mkdirs(sorted({o.parent for _s, o in jobs_c}))
    errors: List[dict] = []
    for src, obj in jobs_c:
        _ok, errs, text = verify.compile_source(
            ninja, bdir, src, obj, [root / FIXTURE_DEST], ex, timeout
        )
        errors += errs
        logs.append(text)
    if not stage("test compiled", not errors, start, "; ".join(e["message"] for e in errors[:3])):
        return finish(
            "build_failed",
            f"{len(errors)} compile error(s) in upstream's reference test: see log.txt",
        )
    start = time.time()
    exe = work / "unittest_wb_selftest"
    is_ok, errs, text = verify.link_objects(
        ninja, bdir, [jobs_c[0][1], utils_obj], exe, ex, timeout, verify.link_extras(ex, gtest=True)
    )
    logs.append(text)
    if not stage("test linked", is_ok, start, "; ".join(e["message"] for e in errs[:3])):
        return finish("build_failed", "the reference test did not link: see log.txt")

    # 6. run its FP32 check
    start = time.time()
    gjson = out / "gtest.json"
    if gjson.exists():
        gjson.unlink()
    env = ex.lib_env(ex.lib_paths())
    env["NNTRAINER_CAUSALLM_FIXTURE_DIR"] = ex.path(root / FIXTURE_DEST / "causallm_reference")
    code, text = ex.run(
        [ex.path(exe), "--gtest_filter=" + TEST_FILTER, "--gtest_output=json:" + ex.live(gjson)],
        bdir,
        timeout,
        env_extra=env,
    )
    logs.append(text)
    test = verify.parse_gtest(gjson).get(TEST_NAME)
    if test is None:
        stage("test run", False, start, f"the test exited with {code} before finishing")
        return finish(
            "crashed", f"the reference test exited with {code} before finishing: see log.txt"
        )
    if test["result"] == "SKIPPED" or "SKIP" in str(test["status"]).upper():
        stage("test run", False, start, "skipped: the fixture was not found")
        return finish(
            "error", "the reference test skipped itself: the tiny Qwen3 fixture was not found"
        )
    passed = code == 0 and not test["failures"]
    detail = (
        ""
        if passed
        else verify.summarize_failure(test["failures"][0]) if test["failures"] else f"exit {code}"
    )
    stage("test run", passed, start, detail)
    if not passed:
        return finish(
            "mismatch", "nntrainer's Qwen3 does not match Hugging Face in Docker: " + detail[:200]
        )
    return finish(
        "passed",
        "the full build, upstream's Qwen3 reference test and its FP32 check against "
        "Hugging Face all passed in Docker",
    )
