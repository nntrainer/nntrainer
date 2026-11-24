# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_models.py
# @brief Tests for the torch and transformers steps, on tiny models built from a config.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
The steps that need torch / transformers / safetensors, on tiny models built
from a config (nothing is downloaded): model discovery from a local folder,
the compatibility step (a known architecture -> semantic IR, an unknown one ->
the module-tree graph, then the cache), the Llama-family adapter (Qwen2 and
Qwen3's per-head q/k norm), the module-tree parser and its dataflow tracing,
the weight download / conversion / preview steps, the meta-device tensor
list, and Verify's fixture script (a tiny model, a generated converter, the
real-weights conversion). Each test returns early when those packages are
missing, so the suite still runs without them.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
from agents import optional_deps  # noqa: E402


def _have(*mods) -> bool:
    return not optional_deps.missing(*mods)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(dir=tg._TMP))


def _events(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, [
        json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")
    ]


def _status(events, agent):
    return [e for e in events if e.get("event") == "agent_status" and e.get("agent") == agent]


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


QWEN2 = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
}
QWEN3 = dict(QWEN2, architectures=["Qwen3ForCausalLM"], model_type="qwen3", head_dim=16)
GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_layer": 1,
    "n_embd": 16,
    "n_head": 2,
    "vocab_size": 64,
    "n_positions": 32,
}


def _folder(cfg) -> Path:
    tmp = _tmp()
    (tmp / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp


def _model(cfg):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.for_model(**cfg)
    return config, AutoModel.from_config(config)


# ------------------------------------------------------- discovery + compat
def test_discovery_and_compatibility_for_a_known_architecture():
    """Discovery and compatibility for a known architecture."""
    if not _have("torch", "transformers"):
        return
    from agents import compatibility, model_discovery
    folder = _folder(QWEN2)
    out = _tmp() / "out"
    state, events = _events(lambda: model_discovery.run({"model_name": str(folder)}))
    assert state["architecture"] == "Qwen2ForCausalLM" and state["hf_config"]["hidden_size"] == 64
    state["out_dir"] = str(out)
    state, events = _events(lambda: compatibility.run(state))
    assert state["semantic_ir"]["num_layers"] == 2 and state["graph_ir"] is None
    assert (
        _status(events, "compatibility")[-1]["detail"]
        == "semantic IR built; module-tree view unavailable"
    )
    again, events = _events(
        lambda: compatibility.run({"model_name": str(folder), "out_dir": str(out)})
    )
    assert _status(events, "compatibility")[-1]["detail"].startswith("semantic IR (cached")
    assert again["semantic_ir"] == state["semantic_ir"]

    bad, events = _events(lambda: model_discovery.run({"model_name": str(_tmp() / "nope")}))
    assert bad["errors"] and _status(events, "model_discovery")[-1]["status"] == "error"
    bad, events = _events(
        lambda: compatibility.run({"model_name": str(_tmp() / "nope"), "out_dir": str(out)})
    )
    assert bad["errors"] and _status(events, "compatibility")[-1]["status"] == "error"


def test_compatibility_for_an_unknown_architecture_traces_the_module_tree():
    """Compatibility for an unknown architecture traces the module tree."""
    if not _have("torch", "transformers"):
        return
    from agents import compatibility
    folder = _folder(GPT2)
    state, events = _events(
        lambda: compatibility.run({"model_name": str(folder), "out_dir": str(_tmp() / "out")})
    )
    assert state.get("semantic_ir") is None and state["graph_ir"]["nodes"]
    summary = state["report"]["summary"]
    assert summary["supported_nodes"] + summary["unsupported_nodes"] == len(
        state["graph_ir"]["nodes"]
    )
    assert _status(events, "compatibility")[-1]["status"] == "done"


# ----------------------------------------------------------------- adapters
def test_the_llama_family_adapter_reads_qwen2_and_qwen3():
    """The Llama family adapter reads Qwen2 and Qwen3."""
    if not _have("torch", "transformers"):
        return
    from api.adapters.registry import select_adapter
    for cfg, qk_norm in ((QWEN2, False), (QWEN3, True)):
        config, model = _model(cfg)
        adapter = select_adapter(config, model)
        assert adapter is not None and cfg["model_type"] in adapter.model_types
        semantic_ir = adapter.build_semantic_ir(config, model)
        assert (
            semantic_ir.num_layers == 2
            and len(semantic_ir.decoder_layers) == 2
            and semantic_ir.hidden_size == 64
        )
        attn = semantic_ir.decoder_layers[0].attention
        assert attn.num_heads == 4 and attn.num_kv_heads == 2
        assert (attn.q_norm is not None) is qk_norm and (not qk_norm or attn.q_norm.reshaped)
        mlp = semantic_ir.decoder_layers[0].mlp
        assert mlp.gated and mlp.up_proj.output_size == 96


# ------------------------------------------------------------- the parser
def test_the_parser_traces_dataflow_or_falls_back_to_a_chain():
    """The parser traces dataflow or falls back to a chain."""
    if not _have("torch"):
        return
    import torch
    from api.parsers.generic_fx_parser import GenericFxParser, TransformerSemanticLoweringRequired

    # pylint: disable-next=too-few-public-methods
    class Toy(torch.nn.Module):
        """A small sequential model whose dataflow the parser can trace."""
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 16)
            self.proj = torch.nn.Linear(16, 16)
            self.act = torch.nn.ReLU()
            self.drop = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(16, 100)

        def forward(self, input_ids=None, **_kwargs):
            """Embedding, projection, activation, dropout and head, in a chain."""
            return self.head(self.drop(self.act(self.proj(self.embed(input_ids)))))
    toy = Toy().eval()
    toy.config = types.SimpleNamespace(model_type="toy", vocab_size=100)
    parser = GenericFxParser(toy, "toy")
    graph = parser.parse()
    names = [n.name for n in graph.get_nodes()]
    assert names[0] == "embed" and "head" in names and parser.edge_source == "traced"
    assert all(n.input_shape is not None for n in graph.get_nodes() if n.name == "proj")

    # pylint: disable-next=too-few-public-methods
    class Residual(Toy):
        """The same model with a residual add that module hooks cannot see."""
        def forward(self, input_ids=None, **_kwargs):
            embed = self.embed(input_ids)
            return self.head(embed + self.proj(embed))  # a functional add hooks cannot see
    res = Residual().eval()
    res.config = types.SimpleNamespace(model_type="toy", vocab_size=100)
    path = GenericFxParser(res, "res")
    graph_ir = path.parse()
    assert path.edge_source == "linear" and len(graph_ir.edges) == len(graph_ir.get_nodes()) - 1

    _config, model = _model(QWEN2)
    try:
        GenericFxParser(model, "qwen2").parse()
        raise AssertionError("a known transformer must not get a declaration-order chain")
    except TransformerSemanticLoweringRequired:
        pass


# ---------------------------------------------------------------- weights
def _safetensors_folder(cfg=None) -> Path:
    cfg = QWEN2 if cfg is None else cfg
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM
    folder = _folder(cfg)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**cfg))
    state = {k: v.contiguous().to(torch.bfloat16 if "mlp" in k else torch.float32)
             for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(folder / "model.safetensors"))
    return folder


def test_weight_download_converter_and_preview():
    """Weight download converter and preview."""
    if not _have("torch", "transformers", "safetensors"):
        return
    from agents import (
        compatibility,
        model_discovery,
        nntrainer_lowering,
        weight_converter,
        weight_download,
    )
    from agents import weight_preview
    folder = _safetensors_folder()
    out = _tmp() / "out"
    new_state, events = _events(
        lambda: weight_download.run(
            {"model_name": "Qwen/x", "out_dir": str(out), "custom_weights_path": str(folder)}
        )
    )
    assert (
        new_state["weights_path"] == str(folder)
        and _status(events, "weight_download")[-1]["detail"] == "custom optimized"
    )
    new_state, events = _events(
        lambda: weight_download.run({"model_name": str(folder), "out_dir": str(out)})
    )
    assert _status(events, "weight_download")[-1]["detail"] == "local path"

    new_state, events = _events(
        lambda: weight_converter.run(
            {"model_name": str(folder), "out_dir": str(out), "weights_path": str(folder)}
        )
    )
    assert os.path.getsize(new_state["converted_weights_path"]) > 0
    assert _status(events, "weight_converter")[-1]["status"] == "done"
    state2, events = _events(
        lambda: weight_converter.run(
            {"model_name": str(folder), "out_dir": str(out), "weights_path": str(folder)}
        )
    )
    assert state2["converted_weights_path"] == new_state["converted_weights_path"]
    assert (
        _events(lambda: weight_converter.run({"out_dir": str(out)}))[0].get(
            "converted_weights_path"
        )
        is None
    )

    state = {"model_name": str(folder), "out_dir": str(out)}
    state, _e = _events(lambda: model_discovery.run(state))
    state, _e = _events(lambda: compatibility.run(state))
    state, _e = _events(lambda: nntrainer_lowering.run(state))
    state["weights_path"] = str(folder)
    state, events = _events(lambda: weight_preview.run(state))
    assert _status(events, "weight_preview")[-1]["status"] == "done"
    previewed = [
        n["weightInfo"]
        for n in state["nntrainer_graph_view"]["nodes"]
        if "mean" in (n.get("weightInfo") or {})
    ]
    assert previewed and all({"min", "max", "std", "preview"} <= set(w) for w in previewed)
    assert any(w["dtype"] == "bfloat16" for w in previewed)  # the real dtype, read from the file
    assert [e["target"] for e in events if e["event"] == "graph"] == ["model", "nntrainer"]
    for missing in ({"weights_path": None}, {"weights_path": str(_tmp())}):
        new_state, events = _events(
            lambda missing=missing: weight_preview.run(dict(state, **missing))
        )
        assert _status(events, "weight_preview")[-1]["status"] == "skipped"
    new_state, events = _events(lambda: weight_preview.run({"out_dir": str(out)}))
    assert _status(events, "weight_preview")[-1]["status"] == "skipped"
    assert weight_preview._finite(float("nan")) is None and weight_preview._finite(1.5) == 1.5


def test_weights_from_the_hub_are_cached():
    """Weights from the Hub are cached."""
    if not _have("torch", "transformers", "safetensors"):
        return
    from agents import weight_download
    out = _tmp() / "out"
    calls = []

    def snapshot(repo_id, **kw):
        calls.append(repo_id)
        dest = Path(kw["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.safetensors").write_bytes(b"x" * 1024)
        return str(dest)
    hub = sys.modules.get("huggingface_hub")
    with _patched(hub, snapshot_download=snapshot):
        state, events = _events(
            lambda: weight_download.run({"model_name": "Org/tiny", "out_dir": str(out)})
        )
        assert state["weights_path"] and calls == ["Org/tiny"]
        state, events = _events(
            lambda: weight_download.run({"model_name": "Org/tiny", "out_dir": str(out)})
        )
        assert calls == ["Org/tiny"] and _status(events, "weight_download")[-1][
            "detail"
        ].startswith("cached")
    assert weight_download._dir_size_mb(state["weights_path"]) > 0


def test_the_tensor_list_from_the_meta_device():
    """The tensor list from the meta device."""
    if not _have("torch", "transformers"):
        return
    from agents import causallm_codegen
    folder = _folder(QWEN2)
    facts, events = _events(lambda: causallm_codegen._facts_from_meta_model(str(folder), QWEN2))
    assert "embed_tokens.weight" in facts.tensors and "lm_head.weight" not in facts.tensors   # tied
    facts, events = _events(
        lambda: causallm_codegen._facts_from_meta_model(str(_tmp() / "nope"), QWEN2)
    )
    assert facts.tensors == {} and any("Could not build" in e.get("message", "") for e in events)


# ------------------------------------------------------- Verify's fixture
CONVERTER = '''
import numpy as np


def save_qwen2_for_nntrainer(params, n_layers, dtype, file):
    for name in sorted(params):
        t = params[name]
        arr = t.detach().float().numpy() if hasattr(t, "detach") else np.asarray(t, \
dtype=np.float32)
        arr.astype(np.float32).tofile(file)
'''


def test_the_verify_fixture_script():
    """The verify fixture script."""
    if not _have("torch", "transformers", "safetensors", "numpy"):
        return
    from knowledge import verify_fixture as vf
    work = _tmp()
    conv = work / "weight_converter.py"
    conv.write_text(CONVERTER, encoding="utf-8")
    cfg = _folder(QWEN2) / "config.json"
    nntr = work / "nntr_config.json"
    nntr.write_text(json.dumps({"model_type": "CausalLM"}), encoding="utf-8")
    out = work / "fixture"
    summary = vf.build(cfg, conv, nntr, out)
    assert (
        summary["ok"]
        and summary["converter_fn"] == "save_qwen2_for_nntrainer"
        and len(summary["tokens"]) > 0
    )
    for name in (
        "config.json",
        "nntr_config.json",
        "reference_logits.json",
        "reference_tokens.json",
        "meta.json",
        "tokenizer.json",
        "nntr_fixture_fp32.bin",
    ):
        assert (out / name).is_file(), name
    assert json.loads((out / "config.json").read_text())["num_hidden_layers"] == 2

    real = vf.convert_real(_safetensors_folder(), conv, work / "real" / "w.bin")
    assert real["ok"] and real["bytes"] > 0 and real["shards"] == 1

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = vf.main(["--converter", str(conv), "--out", str(work / "x")])
    assert code == 1 and "--config and --nntr-config" in json.loads(buf.getvalue())["error"]
    empty = work / "empty.py"
    empty.write_text("x = 1\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = vf.main(["--config", str(cfg), "--converter", str(empty), "--nntr-config", str(nntr),
                        "--out", str(work / "y")])
    assert code == 1 and "save_<model>_for_nntrainer" in json.loads(buf.getvalue())["error"]


def test_the_checklist_reads_tensors_from_transformers_on_the_meta_device():
    """The checklist reads tensors from transformers on the meta device."""
    if not _have("torch", "transformers"):
        return
    from knowledge import checklist as ck
    facts = ck.facts_from_transformers(QWEN3, "qwen3-tiny")
    assert facts.name == "qwen3-tiny" and facts.source == "transformers (meta device)"
    assert facts.tensors["layers.*.self_attn.q_norm.weight"] == [16]  # per head: head_dim
    assert not any(k.endswith("lm_head.weight") for k in facts.tensors)           # tied
    try:
        ck.facts_from_transformers({"hidden_size": 8})
        raise AssertionError("a config without model_type must be refused")
    except ValueError:
        pass
