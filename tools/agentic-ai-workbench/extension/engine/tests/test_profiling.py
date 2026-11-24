# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_profiling.py
# @brief Tests for profiling: the weights lookup, the profiler step and per-layer timing.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
Profiling: the weights lookup (knowledge/run_model.find_model_dir), the
pipeline's profiler step for generated models (agents/profiler_agent.py) and
its place after the weight download (agents/orchestrator.py), and per-layer
timing (the profiling build variant in knowledge/docker_host.py, the
runner's report and run_model.parse_profile). Docker and the runs are faked.
Standard library only.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge import docker_host as dh  # noqa: E402
from knowledge import run_model as rm  # noqa: E402
from knowledge import verify  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(dir=tg._TMP))


def _weights(data: Path) -> Path:
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.json").write_text("{}", encoding="utf-8")
    (data / "model.safetensors").write_bytes(b"")
    return data


# --------------------------------------------------------------- the lookup
def test_the_weights_the_pipeline_downloaded_are_found():
    """The weights the pipeline downloaded are found."""
    wb_dir = _tmp()
    assert (
        rm.short_weights_dir(wb_dir, MODEL)
        == wb_dir / ".w" / hashlib.sha1(MODEL.encode()).hexdigest()[:8]
    )
    assert (
        rm.short_weights_dir(wb_dir, MODEL).name == "028afa86"
    )  # the folder on the development PC
    real = rm.workbench_dir
    rm.workbench_dir = lambda: wb_dir
    try:
        assert rm.find_model_dir({"model": MODEL}) is None
        old = _weights(
            wb_dir / ".w" / hashlib.sha256(MODEL.encode()).hexdigest()[:8]
        )  # 0.4.5-0.4.9's name
        assert rm.find_model_dir({"model": MODEL}) == old
        new = _weights(rm.short_weights_dir(wb_dir, MODEL))  # the pipeline's name wins
        assert rm.find_model_dir({"model": MODEL}) == new
        hint = _weights(_tmp() / "w")
        assert (
            rm.find_model_dir({"model": MODEL}, hint) == hint
        )  # the pipeline's own download first
        assert (
            rm.find_model_dir({"model": MODEL}, _tmp()) == new
        )  # a hint without weights is ignored
    finally:
        rm.workbench_dir = real


# -------------------------------------------------------- per-layer timing
# The runner's PhaseListener report: 1 prompt call + 3 generated tokens.
# layer1_ffn_up is 2.5x slower per token than the same layer in blocks 0 and 2.
REPORT = "\n".join([
    "WORKBENCH_PROFILE_BEGIN",
    "calls\t4",
    "L\tnn_forward\t1\t30000\t3\t90000\t28000\t32000",
    "L\tfully_connected\t1\t20000\t3\t60000\t19000\t21000",
    "L\tlayer0_ffn_up:forward(fully_connected)\t1\t3000\t3\t3600\t1100\t1300",
    "L\tlayer1_ffn_up:forward(fully_connected)\t1\t3000\t3\t9000\t2900\t3100",
    "L\tlayer2_ffn_up:forward(fully_connected)\t1\t3000\t3\t3600\t1100\t1300",
    "L\tlayer0_attention_norm/generated_out_0:forward(multiout)\t1\t1\t3\t3\t1\t1",
    "L\toutput_of_causallm:forward(tie_word_embeddings)\t1\t9000\t3\t30000\t9000\t11000",
    "L\tmha_core\t1\t2000\t3\t3000\t900\t1100",
    "L\tbroken line",
    "WORKBENCH_PROFILE_END", ""])


def _run_line(prefill_ms: float) -> str:
    return (
        "WORKBENCH_RUN "
        + json.dumps(
            {
                "prefill_tokens": 36,
                "prefill_ms": prefill_ms,
                "generation_tokens": 3,
                "generation_ms": 250,
            }
        )
        + "\n"
    )


def test_the_profilers_report_is_read_per_phase():
    """The profilers report is read per phase."""
    profile = rm.parse_profile("noise\n" + _run_line(300) + REPORT + "after\n")
    assert profile["calls"] == 4 and profile["generation_calls"] == 3
    assert (
        profile["forward"]["key"] == "nn_forward"
        and profile["forward"]["generation_avg_us"] == 30000
    )  # not a layer
    assert [t["key"] for t in profile["types"]] == ["fully_connected", "mha_core"]  # biggest first
    first_type = profile["types"][0]
    assert (first_type["prompt_us"], first_type["generation_us"], first_type["total_us"]) == (
        20000,
        60000,
        80000,
    )
    assert (first_type["pct"], first_type["prompt_pct"], first_type["generation_pct"]) == (
        94.12,
        90.91,
        95.24,
    )
    assert profile["totals"] == {"prompt_us": 22000, "generation_us": 63000, "total_us": 85000}
    # single layers, kept apart from the types they overlap with, in the order they ran
    names = [r["name"] for r in profile["layers"]]
    assert names == [
        "layer0_ffn_up",
        "layer1_ffn_up",
        "layer2_ffn_up",
        "layer0_attention_norm/generated_out_0",
        "output_of_causallm",
    ]
    up1 = profile["layers"][1]
    assert (up1["block"], up1["role"], up1["type"], up1["op"]) == (
        1,
        "ffn_up",
        "fully_connected",
        "forward",
    )
    assert (
        up1["generation_avg_us"] == 3000.0
        and up1["generation_min_us"] == 2900
        and up1["total_us"] == 12000
    )
    head = profile["layers"][4]
    assert head["block"] is None and head["pct"] == 60.74                     # 39000 of 64204 us
    assert rm.parse_profile("WORKBENCH_RUN {}\n") is None


def test_the_block_map_finds_the_slow_block():
    """The block map finds the slow block."""
    profile = rm.parse_profile(REPORT)
    block_map = rm.block_map(profile["layers"])
    assert block_map["blocks"] == [0, 1, 2] and block_map["roles"] == [
        "ffn_up"
    ]  # nntrainer's helper layers left out
    assert block_map["grid"] == [[0], [1], [2]]
    assert block_map["outliers"] == {"all": [1], "prompt": [], "generation": [1]}
    assert rm.per_call_us(profile["layers"][1], "all") == 3000  # 12000 us over 4 calls


def _safetensors(path: Path, shapes: dict) -> None:
    header = json.dumps(
        {k: {"dtype": "F32", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_the_memory_speed_of_the_weight_layers():
    """The memory speed of the weight layers."""
    tmp = _tmp()
    _safetensors(
        tmp / "model.safetensors",
        {
            **{f"model.layers.{i}.mlp.up_proj.weight": [4864, 896] for i in range(3)},
            "model.embed_tokens.weight": [151936, 896],
            "model.layers.0.input_layernorm.weight": [896],
        },
    )
    profile = rm.parse_profile(REPORT)
    rm.add_memory_speed(profile, tmp)
    up0, up1, helper, head = (
        profile["layers"][0],
        profile["layers"][1],
        profile["layers"][3],
        profile["layers"][4],
    )
    assert up0["weight_mb"] == 17.4 and up0["gbps"] == 14.5  # 17.4 MB in 1.2 ms per token
    assert up1["gbps"] == 5.8
    assert head["weight_mb"] == 544.5 and head["gbps"] == 54.5  # the tied embedding is the head
    assert "gbps" not in helper
    rm.add_memory_speed(profile, None)  # no checkpoint: nothing breaks


def test_the_runner_reports_layers_only_in_the_profiling_build():
    """The runner reports layers only in the profiling build."""
    src = rm.RUNNER
    assert src.count("#ifdef PROFILE") == 4 and "#include <profiler.h>" in src
    assert "class PhaseListener" in src and "std::make_shared<PhaseListener>()" in src
    assert "WORKBENCH_PROFILE_BEGIN" in src and "listener->report(std::cout)" in src
    code = src.format(cls="Qwen2WorkbenchCausalLM", model="m", header="qwen2_workbench_causallm.h")
    assert "long long n[2] = {0, 0}" in code and 'out << "calls\\t" << calls' in code
    assert 'data->event_str == "nn_forward"' in code


def test_the_profiled_run_comes_after_the_measured_runs_and_retries_a_cold_one():
    """The profiled run comes after the measured runs and retries a cold one."""
    root, run_dir = _tmp(), _tmp()
    weights = _tmp()
    _safetensors(weights / "model.safetensors", {"model.layers.0.mlp.up_proj.weight": [4864, 896]})
    outs = [
        _run_line(5000) + REPORT,
        _run_line(320) + REPORT,
    ]  # the first read the weights from disk
    calls, logged = [], []

    # pylint: disable-next=too-few-public-methods
    class Host:
        """Stand-in Docker host with a volume name."""
        volume = "nntr-wb-x"

    # pylint: disable-next=too-few-public-methods
    class Ex:
        """Stand-in Docker executor for the profiling run."""
        host = Host()

    def fake_stream(args, _timeout, log):
        calls.append(list(args))
        out = outs[len(calls) - 1]
        for line in out.splitlines():
            log(line)
        return 0, out
    with (
        _patched(
            rm,
            profile_executor=lambda r, log: Ex(),
            bundle=lambda review, ex, log, t: (Path("d"), "abc"),
            load_image=lambda review, deploy, digest, log: "nntr-workbench-run:m-prof-abc",
            stage_model=lambda ex, rd, slug, log: "/work/models/m",
        ),
        _patched(dh, stream=fake_stream),
    ):
        prof = rm.profile_layers(
            {"model": MODEL}, root, run_dir, "m", "Hi", "2", None, weights, 300.0, logged.append
        )
    assert len(calls) == 2 and calls[0][-4:] == [
        "nntr-workbench-run:m-prof-abc",
        "/work/models/m",
        "--prompt",
        "Hi",
    ]
    assert "--cpus" in calls[0]
    assert prof["run"]["prefill_ms"] == 320 and prof["image"] == "nntr-workbench-run:m-prof-abc"
    assert prof["map"]["outliers"]["generation"] == [1] and prof["layers"][0]["gbps"] == 14.5
    assert not any(
        line.startswith(("L\t", "calls\t")) for line in logged
    )  # the report stays out of the log
    assert any("Slowest layers: output_of_causallm" in line for line in logged)
    with (
        _patched(
            rm,
            profile_executor=lambda r, log: Ex(),
            bundle=lambda review, ex, log, t: (Path("d"), "abc"),
            load_image=lambda review, deploy, digest, log: "t",
            stage_model=lambda ex, rd, slug, log: "/m",
        ),
        _patched(dh, stream=lambda args, timeout, log: (0, _run_line(300))),
    ):
        assert (
            "no per-layer report"
            in rm.profile_layers(
                {}, root, run_dir, "m", "", None, None, None, 300.0, logged.append
            )["error"]
        )


def test_the_profiling_build_is_a_second_build_of_the_same_copy():
    """The profiling build is a second build of the same copy."""
    root = _tmp()
    normal, prof = dh.DockerHost(root), dh.DockerHost(root, "prof")
    assert normal.name == prof.name and normal.volume == prof.volume  # same container and volume
    assert str(normal.cbuild).endswith("/build-wb") and str(prof.cbuild).endswith("/build-wb-prof")
    assert normal.build_dir.name == "build" and prof.build_dir.name == "build-prof"
    assert prof.to_container(prof.build_dir / "wb_verify" / "x.o").endswith(
        "/build-wb-prof/wb_verify/x.o"
    )
    assert dh.meson_setup_command("prof")[-2:] == ["-Denable-profile=true", str(prof.cbuild)]
    assert "-Denable-profile=true" not in dh.meson_setup_command()
    assert "profiler on" in verify.DockerExecutor(root, "prof").describe()
    assert "profiler" not in verify.DockerExecutor(root).describe()
    try:
        dh.DockerHost(root, "debug")
    except dh.DockerError:
        pass
    else:
        raise AssertionError("expected DockerError")


def test_the_profiling_runner_has_its_own_folders_and_image():
    """The profiling runner has its own folders and image."""
    out = _tmp() / "generated" / "qwen2"
    review = {"out_dir": str(out), "slug": "qwen2"}

    # pylint: disable-next=too-few-public-methods
    class Ex:
        """Stand-in Docker executor with a build variant."""
        def __init__(self, variant=""):
            self.variant = variant
    normal, prof = rm.deploy_dir(review, Ex()), rm.deploy_dir(review, Ex("prof"))
    assert normal.name == "linux-x86_64" and prof.name == "linux-x86_64-prof"
    assert (
        rm.work_dir_for(normal) == out / "run" / "build"
        and rm.work_dir_for(prof) == out / "run" / "build-prof"
    )
    real = dh.image_exists
    dh.image_exists = lambda tag: True
    try:
        assert (
            rm.load_image(review, prof, "abc123", lambda m: None)
            == "nntr-workbench-run:qwen2-prof-abc123"
        )
        assert (
            rm.load_image(review, normal, "abc123", lambda m: None)
            == "nntr-workbench-run:qwen2-abc123"
        )
    finally:
        dh.image_exists = real


def test_run_images_use_the_pinned_base():
    """Run images use the pinned base."""
    assert dh.base_image().startswith("ubuntu:24.04@sha256:")
    assert "FROM {base}" in rm.RUNTIME_DOCKERFILE


def test_clean_up_everything_also_removes_the_profiling_builds_mirror():
    """Clean up everything also removes the profiling build's mirror."""
    root = _tmp()
    wb_dir = root / "tools" / "agentic-ai-workbench"
    for variant in ("build", "build-prof"):
        (wb_dir / "docker" / variant).mkdir(parents=True)
        (wb_dir / "docker" / variant / "build.ninja").write_text("", encoding="utf-8")
    disk_usage = {
        "Images": [],
        "Volumes": [{"Name": dh.DockerHost(root).volume, "Size": "1GB"}],
        "Containers": [],
    }
    cleanup = dh.cleanup(
        root,
        "all",
        lambda args, t: (
            (0, json.dumps(disk_usage)) if list(args[:2]) == ["system", "df"] else (0, "")
        ),
    )
    assert (
        cleanup["ok"]
        and not (wb_dir / "docker" / "build").exists()
        and not (wb_dir / "docker" / "build-prof").exists()
    )


# ----------------------------------------------------- the pipeline profiler
def _events(func):
    from agents.events import bus
    got = []
    for name in ("log", "agent_status", "profile", "codegen"):
        bus.subscribe(name, got.append)
    start = len(got)
    with contextlib.redirect_stdout(io.StringIO()):
        out = func()
    return out, got[start:]


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, attr in attrs.items():
        setattr(obj, k, attr)
    try:
        yield
    finally:
        for k, attr in old.items():
            setattr(obj, k, attr)


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, val in kv.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    try:
        yield
    finally:
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def _verified_review():
    out = _tmp() / "generated" / "m"
    out.mkdir(parents=True)
    path = out / "review.json"
    review = {"model": MODEL, "out_dir": str(out), "review_path": str(path), "files": {},
              "verify": {"status": "passed", "files_hash": verify.verify_hash({})}}
    path.write_text(json.dumps(review), encoding="utf-8")
    return path, review


def test_the_pipeline_profiles_a_verified_model_with_its_real_weights():
    """The pipeline profiles a verified model with its real weights."""
    from agents import profiler_agent, codegen_review
    path, review = _verified_review()
    weights = _weights(_tmp() / "weights")
    calls = []
    record = {
        "ok": True,
        "device": "this PC",
        "summary": {"generation_tps": 12.6, "prefill_tps": 119.0},
        "peak_memory_kb": 2542996,
        "runs": [{"generation_tps": 12.6}],
        "text": "Paris.",
        "error": "",
    }

    def fake_run(_review_path, **kw):
        calls.append(kw)
        return dict(review, run=record)
    with (
        _patched(cg, load_review=lambda p: dict(review)),
        _patched(dh, daemon_status=lambda: (True, "Docker 29")),
        _patched(rm, run_generated=fake_run),
        _patched(codegen_review, docker_ready=lambda kind, step: None),
        _env(
            WORKBENCH_AUTO_PROFILE=None,
            WORKBENCH_RUN_TOKENS="32",
            WORKBENCH_RUN_REPEATS="2",
            WORKBENCH_RUN_CPUS="2",
            WORKBENCH_RUN_MEMORY=None,
        ),
    ):
        state, events = _events(lambda: profiler_agent.run({"codegen_review_path": str(path),
                                                            "weights_path": str(weights)}))
    assert state["profile"]["ran"] and state["profile"]["summary"]["generation_tps"] == 12.6
    assert calls == [
        {
            "prompt": profiler_agent.DEFAULT_PROMPT,
            "tokens": 32,
            "repeat": 2,
            "cpus": "2",
            "memory": None,
            "log": calls[0]["log"],
            "model_dir": weights,
            "per_layer": True,
        }
    ]
    with (
        _patched(cg, load_review=lambda p: dict(review)),
        _patched(dh, daemon_status=lambda: (True, "Docker 29")),
        _patched(rm, run_generated=fake_run),
        _patched(codegen_review, docker_ready=lambda kind, step: None),
        _env(WORKBENCH_AUTO_PROFILE=None, WORKBENCH_AUTO_PROFILE_PER_LAYER="0"),
    ):
        _events(
            lambda: profiler_agent.run(
                {"codegen_review_path": str(path), "weights_path": str(weights)}
            )
        )
    assert calls[-1]["per_layer"] is False  # aiCompilerWorkbench.autoProfilePerLayer off
    profiles = [e for e in events if e["event"] == "profile"]
    assert profiles and profiles[0]["data"]["kind"] == "llm_run" and profiles[0]["data"]["ran"]
    done = [e for e in events if e["event"] == "agent_status" and e.get("status") == "done"]
    assert done and "12.6 tok/s generation" in done[-1]["detail"]
    assert not any("Nothing was compiled" in e.get("message", "") for e in events)


def test_the_pipeline_says_why_it_did_not_profile():
    """The pipeline says why it did not profile."""
    from agents import profiler_agent
    path, review = _verified_review()
    weights = _weights(_tmp() / "weights")
    state = {"codegen_review_path": str(path), "weights_path": str(weights)}

    def reason(**env):
        with _env(**env):
            new_state, _ = _events(lambda: profiler_agent.run(dict(state)))
        assert not new_state["profile"]["ran"]
        return new_state["profile"]["reason"]
    with (
        _patched(cg, load_review=lambda p: dict(review)),
        _patched(dh, daemon_status=lambda: (True, "Docker 29")),
    ):
        assert "autoProfile" in reason(WORKBENCH_AUTO_PROFILE="0")
    unverified = dict(review, verify={"status": "mismatch"})
    with (
        _patched(cg, load_review=lambda p: dict(unverified)),
        _patched(dh, daemon_status=lambda: (True, "")),
    ):
        assert "Verify has not passed" in reason(WORKBENCH_AUTO_PROFILE=None)
    with (
        _patched(cg, load_review=lambda p: dict(review)),
        _patched(dh, daemon_status=lambda: (False, "")),
    ):
        assert "Docker is not running" in reason(WORKBENCH_AUTO_PROFILE=None)
    with (
        _patched(cg, load_review=lambda p: dict(review)),
        _patched(dh, daemon_status=lambda: (True, "")),
        _patched(rm, find_model_dir=lambda r, hint=None: None),
    ):
        assert "not on this computer" in reason(WORKBENCH_AUTO_PROFILE=None)
    new_state, _e = _events(
        lambda: __import__("agents.profiler_agent", fromlist=["run"]).run({})
    )  # the generic path (no generated model)
    assert new_state["profile"]["ran"] is False


def test_the_pipeline_waits_for_the_weights_before_profiling():
    """The pipeline waits for the weights before profiling."""
    from agents import orchestrator as o
    order = []

    def step(name):
        def step_fn(state):
            order.append(name)
            return state
        return step_fn
    mods = {"model_discovery": o.model_discovery, "compatibility": o.compatibility,
            "nntrainer_lowering": o.nntrainer_lowering, "ini_generator": o.ini_generator,
            "graph_builder": o.graph_builder, "cpp_generator": o.cpp_generator_agent,
            "causallm_codegen": o.causallm_codegen, "causallm_install": o.causallm_install,
            "dual_graph": o.dual_graph, "compiler": o.compiler_agent, "profiler": o.profiler_agent,
            "weight_preview": o.weight_preview, "artifact_manager": o.artifact_manager}
    real = {name: m.run for name, m in mods.items()}
    real_start, real_join = o.start_weight_download, o.join_weight_download
    try:
        for name, mod in mods.items():
            mod.run = step(name)
        o.start_weight_download = step("start_weight_download")
        o.join_weight_download = step("join_weight_download")
        with contextlib.redirect_stdout(io.StringIO()):
            o._run_sequential({"compiled": True, "errors": []})
    finally:
        for name, mod in mods.items():
            mod.run = real[name]
        o.start_weight_download, o.join_weight_download = real_start, real_join
    assert (
        order.index("join_weight_download")
        < order.index("profiler")
        < order.index("weight_preview")
    )
    assert order[-1] == "artifact_manager"


# ----------------------------------------------------- On-Device Profiling
def test_the_on_device_harness_runs_the_generated_model():
    """The on device harness runs the generated model."""
    # The harness's main() must build the model the template generator wrote,
    # not an empty one from nntrainer's createModel() factory.
    from agents import device_profiler as dp
    from api.graph.graph import Graph
    from converters.cpp_generator import CPPGenerator
    code = CPPGenerator(Graph()).generate()
    builder = re.search(r"std::unique_ptr<Model> (\w+)\(\)", code).group(1)
    tmp = _tmp()
    cpp, harness = tmp / "generated_model.cpp", tmp / "harness.cpp"
    cpp.write_text(
        code + "\n#ifdef NNTRAINER_STANDALONE_SMOKE_TEST\nint main() { return 0; }\n#endif\n",
        encoding="utf-8",
    )
    dp._write_harness(str(cpp), str(harness), str(tmp / "report.json"), "")
    text = harness.read_text(encoding="utf-8")
    assert f"auto model = {builder}();" in text
    assert text.count("int main()") == 1  # the smoke-test main() is gone
