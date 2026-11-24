# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_graph_pipeline.py
# @brief Tests for the graph and code-generation steps on small hand-built inputs.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
The graph and code-generation steps of the pipeline, run for real on small
hand-built inputs (no torch, no model download):
- an unknown architecture: a module-tree graph -> model.ini -> the template
  generator's generated_model.cpp -> both graph panes parsed back from those
  files, with click-to-highlight;
- a known architecture: a semantic IR -> lowering -> the CausalLM component
  (.h/.cpp + manifests) -> both panes from the IR;
plus the compatibility cache, the op-level report, the semantic feature report,
the adapter registry and the compiler step's Verify. Standard library only.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
import test_compile_and_chat as tcc  # noqa: E402
import test_semantic_lowering as tsl  # noqa: E402
from agents import (
    compatibility,
    compiler_agent,
    cpp_generator_agent,
    dual_graph,
    graph_builder,
)  # noqa: E402
from agents import graph_views, ini_generator, nntrainer_lowering  # noqa: E402
from converters.cpp_generator import CPPGenerator  # noqa: E402

from api.adapters import registry  # noqa: E402
from api.adapters.base import ArchitectureAdapter  # noqa: E402
from api.compatibility import semantic_capabilities  # noqa: E402
from api.compatibility.op_level_checker import OpLevelCompatibilityChecker  # noqa: E402
from api.graph.graph import Graph  # noqa: E402
from api.graph.node import GraphNode  # noqa: E402
from api.lowering.nntrainer.validation import Diagnostic, ValidationResult  # noqa: E402
from api.semantic.model import CausalLMIR  # noqa: E402


def _tmp() -> str:
    return tempfile.mkdtemp(dir=tg._TMP)


def _events(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, [
        json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")
    ]


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


def _status(events, agent):
    return [e for e in events if e.get("event") == "agent_status" and e.get("agent") == agent]


# --------------------------------------------------- an unknown architecture
def _module_tree_graph() -> Graph:
    """embeddings -> dropout (passthrough) -> dense -> rotary (unsupported) -> head."""
    graph = Graph()
    graph.model_name, graph.architecture = "toy/model", "ToyForMaskedLM"
    nodes = [
        GraphNode(
            name="embeddings.word_embeddings",
            node_type="embedding",
            id="n1",
            supported=True,
            attributes={"in_dim": 100, "out_dim": 16},
            weight_name="embeddings.word_embeddings.weight",
            weight_shape=(100, 16),
            weight_dtype="float32",
            parameter_count=1600,
            input_shape=(1, 4),
            output_shape=(1, 4, 16),
        ),
        GraphNode(
            name="embeddings.dropout",
            node_type="",
            id="n2",
            supported=False,
            status="passthrough",
            compatibility_reason="no compute",
        ),
        GraphNode(
            name="encoder.dense",
            node_type="fully_connected",
            id="n3",
            supported=True,
            attributes={"unit": 16},
            weight_name="encoder.dense.weight",
            weight_shape=(16, 16),
            weight_dtype="float32",
            parameter_count=272,
        ),
        GraphNode(
            name="encoder.rotary",
            node_type="",
            id="n4",
            supported=False,
            compatibility_reason="no nntrainer equivalent",
        ),
        GraphNode(
            name="head",
            node_type="fully_connected",
            id="n5",
            supported=True,
            attributes={"unit": 100},
        ),
    ]
    for node in nodes:
        graph.add_node(node)
    for src, dst in zip(nodes, nodes[1:]):
        graph.connect(src, dst)
    return graph


def test_an_unknown_architecture_goes_from_module_tree_to_both_panes():
    """An unknown architecture goes from module tree to both panes."""
    module_tree_graph = _module_tree_graph()
    out = _tmp()
    state = {"graph_ir": module_tree_graph.export(), "out_dir": out, "model_name": "toy/model",
             "suggestions": {"encoder.rotary": "write a custom rotary layer"}}

    state, events = _events(lambda: ini_generator.run(state))
    ini = state["ini_content"]
    assert "[embeddings_word_embeddings]" in ini and "[encoder_dense]" in ini and "[head]" in ini
    assert "[embeddings_dropout]" not in ini and "[encoder_rotary]" not in ini
    sections = {block.split("\n", 1)[0]: block for block in ini.split("\n\n")}
    assert (
        "input_layers = embeddings_word_embeddings" in sections["[encoder_dense]"]
    )  # through the dropout
    assert (
        "input_layers = encoder_dense" in sections["[head]"]
    )  # ... and through the unsupported op
    assert os.path.exists(state["ini_path"])

    state, events = _events(lambda: graph_builder.run(state))
    assert (
        len(state["graph_view"]["nodes"]) == 5
        and _status(events, "graph_builder")[-1]["status"] == "done"
    )
    assert {n["status"] for n in state["graph_view"]["nodes"]} == {"mapped", "unmapped"}

    state, events = _events(lambda: cpp_generator_agent.run(state))
    code = state["cpp_code"]
    assert state["cpp_emission_mode"] == "model_api" and os.path.exists(state["cpp_path"])
    assert (
        "std::unique_ptr<Model> buildModel()" in code and "NNTRAINER_STANDALONE_SMOKE_TEST" in code
    )
    assert "// TODO(unsupported): encoder.rotary" in code
    assert "// Suggested approach: write a custom rotary layer" in code
    assert 'setProperty({"input_layers=embeddings.word_embeddings"})' in code

    state, events = _events(lambda: dual_graph.run(state))
    targets = [e["target"] for e in events if e["event"] == "graph"]
    assert targets == ["model", "nntrainer"]
    files = {(e["target"], e["filename"]) for e in events if e["event"] == "file_content"}
    assert files == {("model", "model.ini"), ("nntrainer", "generated_model.cpp")}
    model_ids = {n["id"] for n in state["model_graph_view"]["nodes"]}
    assert {"embeddings_word_embeddings", "encoder_dense", "head"} <= model_ids
    weighted = [n for n in state["nntrainer_graph_view"]["nodes"] if n.get("weightInfo")]
    assert weighted, "weights parsed from the // Weight comments"
    assert state["node_mappings"] and state["ini_graph_view"] is state["model_graph_view"]

    # The op-level report and a generator round trip on the same graph.
    report = OpLevelCompatibilityChecker().analyze(module_tree_graph)
    assert report["summary"] == {
        "supported_nodes": 3,
        "unsupported_nodes": 2,
        "compatibility": 60.0,
    }
    assert [n["name"] for n in report["unsupported"]] == ["embeddings.dropout", "encoder.rotary"]
    path = os.path.join(out, "saved.cpp")
    CPPGenerator(module_tree_graph).save(path)
    with open(path, encoding="utf-8") as handle:
        assert "buildModel()" in handle.read()


def test_steps_without_a_graph_say_so():
    """Steps without a graph say so."""
    for step, agent in ((ini_generator.run, "ini_generator"), (graph_builder.run, "graph_builder"),
                        (cpp_generator_agent.run, "cpp_generator")):
        _, events = _events(lambda step=step: step({"out_dir": _tmp()}))
        assert _status(events, agent)[-1]["status"] == "error"
    _, events = _events(lambda: nntrainer_lowering.run({}))
    assert _status(events, "nntrainer_lowering")[-1]["detail"] == "skipped"
    _, events = _events(lambda: dual_graph.run({}))
    assert any("skipping Model Graph" in e.get("message", "") for e in events)


# ----------------------------------------------------- a known architecture
def _semantic_state(model_ir: CausalLMIR):
    return {
        "semantic_ir": model_ir.to_dict(),
        "hf_config": {"torch_dtype": "bfloat16"},
        "out_dir": _tmp(),
        "model_name": "Qwen/Qwen3-test",
    }


def test_a_known_architecture_is_lowered_generated_and_shown():
    """A known architecture is lowered generated and shown."""
    state = _semantic_state(tsl._make_model_ir(num_layers=3))
    state, events = _events(lambda: nntrainer_lowering.run(state))
    assert _status(events, "nntrainer_lowering")[-1]["status"] == "done"
    assert state["nntrainer_graph_ir"]["nodes"] and state["node_mappings"]
    assert state["lowering_diagnostics"]["ok"] is True
    dtypes = {
        (n.get("weightInfo") or {}).get("dtype") for n in state["nntrainer_graph_view"]["nodes"]
    }
    assert "bfloat16" in dtypes  # the checkpoint's declared dtype

    state, events = _events(lambda: ini_generator.run(state))
    assert _status(events, "ini_generator")[-1]["detail"] == "skipped (causallm_component)"

    state, events = _events(lambda: cpp_generator_agent.run(state))
    assert state["cpp_emission_mode"] == "causallm_component"
    for key in ("causallm_header_path", "causallm_source_path"):
        assert os.path.exists(state[key])
    folder = os.path.dirname(state["causallm_header_path"])
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
        assert json.load(handle)["num_hidden_layers"] == 3
    with open(os.path.join(folder, "weight_manifest.json"), encoding="utf-8") as handle:
        assert json.load(handle)
    assert state["model_metadata"] and state["weight_manifest"]

    state, events = _events(lambda: dual_graph.run(state))
    names = sorted(e["filename"] for e in events if e["event"] == "file_content")
    assert names == sorted(
        os.path.basename(state[k]) for k in ("causallm_header_path", "causallm_source_path")
    )
    assert [e["target"] for e in events if e["event"] == "graph"] == ["model", "nntrainer"]
    assert any(e["event"] == "node_mappings" and e["mappings"] for e in events)


def test_non_uniform_layers_and_lowering_errors_are_reported():
    """Non uniform layers and lowering errors are reported."""
    semantic_ir = tsl._make_model_ir(num_layers=2)
    semantic_ir.decoder_layers[1] = tsl._make_layer(
        1, qk_norm=False
    )  # layer 1 differs from layer 0
    state = _semantic_state(semantic_ir)
    state, _e = _events(lambda: nntrainer_lowering.run(state))
    state, events = _events(lambda: cpp_generator_agent.run(state))
    assert _status(events, "cpp_generator")[-1]["status"] == "error"
    assert any("component generation failed" in e.get("message", "") for e in events)

    real = nntrainer_lowering._lower_and_validate

    def broken(model_ir):
        graph, _diag = real(model_ir)
        return graph, ValidationResult([Diagnostic("error", 0, "q_proj wired to the wrong input")])
    with _patched(nntrainer_lowering, _lower_and_validate=broken):
        state, events = _events(
            lambda: nntrainer_lowering.run(_semantic_state(tsl._make_model_ir()))
        )
    assert _status(events, "nntrainer_lowering")[-1]["status"] == "error"
    assert state["errors"] == ["nntrainer_lowering: q_proj wired to the wrong input"]


def test_semantic_feature_report():
    """Semantic feature report."""
    model_ir = tsl._make_model_ir()
    rep = semantic_capabilities.describe(model_ir)
    assert rep["attention"]["grouped_query_attention"].startswith("supported")
    assert rep["attention"]["qk_norm"].startswith("custom registered layer")
    assert rep["mlp"]["activation"] == "silu"
    no_layers = CausalLMIR(
        architecture="x",
        hidden_size=1,
        vocab_size=1,
        num_layers=0,
        embedding_name="e",
        decoder_layers=[],
        final_norm=None,
        lm_head_name="h",
        tied_embeddings=True,
    )
    assert semantic_capabilities.describe(no_layers) == {}


# ------------------------------------------------------------ compatibility
def test_compatibility_cache_round_trip_and_suggestions():
    """Compatibility cache round trip and suggestions."""
    root = os.path.join(_tmp(), "cache")
    graph_ir, report = {"nodes": [{"id": 1}]}, {
        "summary": {"compatibility": 50.0},
        "unsupported": [],
    }
    compatibility._save_cached_graph(
        root, graph_ir, report, {"architecture": "qwen3"}, {"attention": {}}
    )
    loaded = compatibility._load_cached_graph(root)
    assert loaded == (graph_ir, report, {"architecture": "qwen3"}, {"attention": {}})
    meta = compatibility.cache.read_meta(root)["graph"]
    assert meta["nodes"] == 1 and meta["has_semantic_ir"] is True
    assert compatibility._load_cached_graph(_tmp()) is None
    with open(os.path.join(root, "report.json"), "w", encoding="utf-8") as handle:
        handle.write("{broken")
    assert compatibility._load_cached_graph(root) is None

    state = {}
    compatibility._maybe_suggest(state, [])
    assert "suggestions" not in state
    _r, events = _events(lambda: compatibility._maybe_suggest(state, [{"name": "x"}]))
    assert "Set Anthropic API Key" in events[0]["message"] and "suggestions" not in state
    with _patched(
        compatibility, _get_llm_suggestions=lambda key, ops: {o["name"]: "do y" for o in ops}
    ):
        compatibility._maybe_suggest({"anthropic_api_key": "k"}, [{"name": "x"}])
        new_state = {"anthropic_api_key": "k"}
        compatibility._maybe_suggest(new_state, [{"name": "x"}])
    assert new_state["suggestions"] == {"x": "do y"}
    from agents import optional_deps
    if optional_deps.missing("langchain_anthropic"):
        out, events = _events(lambda: compatibility._get_llm_suggestions("k", [{"name": "x"}]))
        assert out == {} and "not installed" in events[0]["message"]


# ----------------------------------------------------------------- adapters
# pylint: disable-next=too-few-public-methods
class _Adapter(ArchitectureAdapter):
    model_types = ("toyarch",)

    @staticmethod
    def build_semantic_ir(_config, _model):
        """No IR: the test only checks the adapter is picked."""
        return None


class _Broken(ArchitectureAdapter):
    model_types = ("broken",)

    @staticmethod
    def matches(config, model):
        """Fail, as a misbehaving adapter would."""
        raise RuntimeError("bad adapter")

    @staticmethod
    def build_semantic_ir(_config, _model):
        """Never reached: matches() fails first."""
        return None


def test_the_adapter_registry_picks_the_first_match_and_skips_broken_adapters():
    """The adapter registry picks the first match and skips broken adapters."""
    cfg = type("Cfg", (), {"model_type": "toyarch"})()
    assert registry.select_adapter(cfg, None) is None
    added = [_Broken(), _Adapter()]
    for added_item in added:
        registry.register(added_item)
    try:
        assert registry.select_adapter(cfg, None) is added[1]
        assert repr(added[1]) == "_Adapter(model_types=('toyarch',))"
        assert added[1].matches(cfg, None) and added[1].build_semantic_ir(cfg, None) is None
        qwen = registry.select_adapter(type("Cfg", (), {"model_type": "qwen3"})(), None)
        assert qwen is not None and "qwen3" in qwen.model_types
    finally:
        for added_item in added:
            registry._ADAPTERS.remove(added_item)


# ------------------------------------------------ the compiler step's Verify
def test_the_compiler_step_verifies_only_when_it_can():
    """The compiler step verifies only when it can."""
    from knowledge import preflight, verify
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    path = review["review_path"]
    step = lambda: _events(lambda: compiler_agent._verify_step({}, path))  # noqa: E731
    with tcc._env(WORKBENCH_CODE_BANK=str(bank.path), WORKBENCH_AUTO_VERIFY="0"):
        assert _status(step()[1], "compiler")[-1]["detail"] == "compile check passed"
    with tcc._env(WORKBENCH_CODE_BANK=str(bank.path), WORKBENCH_AUTO_VERIFY=None):
        with _patched(verify, build_status=lambda: {"ok": False}):
            state, events = step()
        assert "verify: set up build" in _status(events, "compiler")[-1]["detail"]
        with (
            _patched(verify, build_status=lambda: {"ok": True, "host": "docker"}),
            _patched(
                preflight,
                gate=lambda step, log, emit: {"ok": False, "summary": "Docker is not running"},
            ),
        ):
            state, events = step()
        assert "Docker is not running" in _status(events, "compiler")[-1]["detail"]

        def fails(*a, **k):
            raise verify.VerifyError("no build")
        with _patched(
            verify, build_status=lambda: {"ok": True, "host": "native"}, verify_generated=fails
        ):
            state, events = step()
        assert (
            _status(events, "compiler")[-1]["detail"] == "compile check passed (verify did not run)"
        )

        passed = dict(
            review,
            verify={
                "status": "passed",
                "ok": True,
                "max_abs_diff": 0.001,
                "atol": 0.01,
                "marked_verified": ["x"],
                "log": "long log",
            },
        )
        with _patched(verify, build_status=lambda: {"ok": True, "host": "native"},
                      verify_generated=lambda *a, **k: passed):
            state, events = step()
        assert state["verify"]["ok"] and "log" not in state["verify"]
        assert _status(events, "compiler")[-1]["detail"] == "verified vs Hugging Face"
        mismatch = dict(review, verify={"status": "mismatch", "ok": False, "failures": []})
        with _patched(verify, build_status=lambda: {"ok": True, "host": "native"},
                      verify_generated=lambda *a, **k: mismatch):
            state, events = step()
        assert _status(events, "compiler")[-1]["status"] == "error"
        # An earlier pass for the same files is reused without building again.
        saved = cg_save_verify(path, {"status": "passed", "max_abs_diff": 0.002,
                                      "files_hash": verify.current_verify_hash(review)})
        state, events = step()
        assert any("Already verified" in e.get("message", "") for e in events) and saved


def cg_save_verify(path, verify_res):
    """Write a verify result into a review.json, as Verify does."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data["verify"] = verify_res
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return True


def test_graph_views_parse_weight_comments():
    """Graph views parse weight comments."""
    assert graph_views._parse_weight_comment(
        "// Weight: name=a.weight shape=[16, 4] dtype=float32 params=64") == {
        "name": "a.weight", "shape": [16, 4], "dtype": "float32", "params": 64}
    assert graph_views._parse_weight_comment("// nothing here") is None
