# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_code_bank.py
# @brief Tests for the code bank and the C++ reader used by the capture step.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position

"""
Code bank (knowledge/code_bank.py) and the C++ reader the capture step uses
(knowledge/cpp_extract.py): the reuse rule, the approve -> verify life cycle,
keyword search, the optional embedding re-rank, and C++ extraction.
Standard library only, like the other engine tests.
"""
import atexit
import os
import shutil
import sqlite3
import sys
import tempfile

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from knowledge import code_bank as cb  # noqa: E402
from knowledge.code_bank import CodeBank, CodeBankError  # noqa: E402
from knowledge import cpp_extract as cx  # noqa: E402

# Every test bank lives under one temp folder that is removed at exit
# (connections are closed first; Windows cannot delete an open database).
_TMP = tempfile.mkdtemp(prefix="codebank_test_")
_OPEN = []


def _cleanup():
    for bank in _OPEN:
        try:
            bank.close()
        except sqlite3.Error:
            pass
    shutil.rmtree(_TMP, ignore_errors=True)


atexit.register(_cleanup)


def _bank():
    bank = CodeBank(os.path.join(tempfile.mkdtemp(dir=_TMP), "bank.db"))
    _OPEN.append(bank)
    return bank


def _hook(id_, features, status="approved", supports=(), code="", **kw):
    entry = {
        "id": id_,
        "kind": "hook",
        "name": id_,
        "hook": "createAttention",
        "features": list(features),
        "supports": list(supports),
        "status": status,
        "code": code or f"// {id_}",
    }
    if status == "verified":
        entry["verification"] = {"method": "test"}
    entry.update(kw)
    return entry


# ------------------------------------------------------------- reuse rule
def test_only_verified_entries_are_reusable():
    """Only verified entries are reusable."""
    bank = _bank()
    bank.put(_hook("hook:a", ["attn", "qk_norm"], status="verified"))
    bank.put(_hook("hook:b", ["attn", "qk_norm"], status="approved"))
    assert [
        e["id"] for e in bank.find_reusable("hook", ["attn", "qk_norm"], hook="createAttention")
    ] == ["hook:a"]
    matching = bank.find_matching("hook", ["attn", "qk_norm"], hook="createAttention")
    assert {e["id"] for e in matching} == {"hook:a", "hook:b"}


def test_feature_match_needs_required_and_allows_supported_only():
    """Feature match needs required and allows supported only."""
    bank = _bank()
    bank.put(_hook("hook:base", ["attn"], status="verified", supports=["window"]))
    assert bank.find_reusable("hook", ["attn"])                     # required only
    assert bank.find_reusable("hook", ["attn", "window"])           # plus a supported one
    assert not bank.find_reusable("hook", ["attn", "sinks"])        # something it does not handle
    assert not bank.find_reusable("hook", ["window"])               # missing what it requires


def test_exact_feature_match_ranks_first():
    """Exact feature match ranks first."""
    bank = _bank()
    bank.put(_hook("hook:wide", ["attn"], status="verified", supports=["qk_norm", "window"]))
    bank.put(_hook("hook:exact", ["attn", "qk_norm"], status="verified"))
    assert bank.find_reusable("hook", ["attn", "qk_norm"])[0]["id"] == "hook:exact"


def test_verified_needs_evidence():
    """Verified needs evidence."""
    bank = _bank()
    hook = _hook("hook:x", ["attn"], status="verified")
    hook["verification"] = {}
    try:
        bank.put(hook)
    except CodeBankError:
        return
    raise AssertionError("a verified entry without evidence was accepted")


# -------------------------------------------------------------- life cycle
def test_generated_to_approved_to_verified():
    """Generated to approved to verified."""
    bank = _bank()
    bank.add_generated(_hook("hook:new", ["attn", "post_norm"]))
    assert bank.get("hook:new")["status"] == "generated"
    assert not bank.find_matching("hook", ["attn", "post_norm"])    # not even an example yet
    bank.approve("hook:new", actor="reviewer", notes="matches HF forward")
    assert bank.find_matching("hook", ["attn", "post_norm"])[0]["status"] == "approved"
    assert not bank.find_reusable("hook", ["attn", "post_norm"])
    bank.mark_verified("hook:new", {"method": "hf-logits", "max_abs_diff": 0.004})
    hook_new = bank.get("hook:new")
    assert hook_new["status"] == "verified" and hook_new["approved_by"] == "reviewer"
    assert bank.find_reusable("hook", ["attn", "post_norm"])[0]["id"] == "hook:new"
    actions = [h["action"] for h in bank.history("hook:new")]
    assert actions == ["add", "approve", "verify"]


def test_cannot_verify_without_approval():
    """Cannot verify without approval."""
    bank = _bank()
    bank.add_generated(_hook("hook:new", ["attn"]))
    try:
        bank.mark_verified("hook:new", {"method": "hf-logits"})
    except CodeBankError:
        return
    raise AssertionError("verified an entry nobody approved")


def test_failed_verification_stops_reuse_until_fixed_and_reverified():
    """Failed verification stops reuse until fixed and reverified."""
    bank = _bank()
    bank.add_generated(_hook("hook:new", ["attn"]))
    bank.approve("hook:new", actor="reviewer")
    bank.mark_verified("hook:new", {"method": "hf-logits"})
    bank.mark_failed("hook:new", {"method": "hf-logits", "max_abs_diff": 3.1})
    assert not bank.find_reusable("hook", ["attn"])
    assert not bank.find_matching("hook", ["attn"])  # not offered as an example either
    bank.approve("hook:new", actor="reviewer", notes="fixed rope")
    bank.mark_verified("hook:new", {"method": "hf-logits"})
    assert bank.find_reusable("hook", ["attn"])


def test_withdrawn_and_rejected_are_never_returned():
    """Withdrawn and rejected are never returned."""
    bank = _bank()
    bank.put(_hook("hook:v", ["attn"], status="verified"))
    bank.withdraw("hook:v", actor="reviewer", reason="wrong for long context")
    bank.add_generated(_hook("hook:g", ["attn"]))
    bank.reject("hook:g", actor="reviewer")
    assert not bank.find_matching("hook", ["attn"], statuses=cb.STATUSES[:3])
    assert bank.search("attn", embeddings="off") == []


def test_recapture_of_unchanged_entry_is_a_no_op():
    """Recapture of unchanged entry is a no op."""
    bank = _bank()
    hook = _hook("hook:x", ["attn"], status="verified")
    assert bank.put(hook) == "added"
    assert bank.put(dict(hook)) == "unchanged"
    assert bank.put(dict(hook, code="// changed")) == "updated"


# ------------------------------------------------------------------ search
def test_keyword_search_splits_identifiers():
    """Keyword search splits identifiers."""
    bank = _bank()
    bank.put(
        _hook(
            "hook:Qwen3Transformer::createAttention",
            ["attn", "qk_norm_per_head"],
            status="verified",
            code='LayerHandle q_norm(createLayer("reshaped_rms_norm", {}));',
        )
    )
    bank.put(
        _hook(
            "hook:Transformer::createMlp",
            ["mlp_gated_silu"],
            status="verified",
            hook="createMlp",
            code='createLayer("swiglu", {})',
        )
    )
    ids = [h["id"] for h in bank.search("rms_norm attention", embeddings="off")]
    assert ids and ids[0] == "hook:Qwen3Transformer::createAttention"
    assert [h["id"] for h in bank.search("swiglu", hook="createMlp", embeddings="off")] == [
        "hook:Transformer::createMlp"
    ]


def test_search_examples_exclude_unapproved_by_default():
    """Search examples exclude unapproved by default."""
    bank = _bank()
    bank.add_generated(_hook("hook:g", ["attn"], code="sliding window attention"))
    bank.put(_hook("hook:a", ["attn"], code="sliding window attention"))
    assert [h["id"] for h in bank.search("sliding window", embeddings="off")] == ["hook:a"]


# pylint: disable-next=too-few-public-methods
class _FakeEmbedder(cb.Embedder):
    model = "fake"

    @staticmethod
    def embed(texts):
        """Two-dimensional vectors: texts about rope point one way, the rest the other."""
        return [[1.0, 0.0] if "rope" in t.lower() else [0.0, 1.0] for t in texts]


def test_embeddings_rerank_keyword_hits_when_available():
    """Embeddings rerank keyword hits when available."""
    bank = _bank()
    bank.put(_hook("hook:a", ["attn"], code="attention attention attention"))
    bank.put(_hook("hook:b", ["attn"], code="attention rope"))
    cb._EMBEDDER_CACHE["fake"] = _FakeEmbedder()
    try:
        hits = bank.search("rope attention", embeddings="fake")
    finally:
        cb._EMBEDDER_CACHE.pop("fake", None)
    assert hits[0]["id"] == "hook:b" and hits[0]["match"] == "keyword+embedding"
    assert bank.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2   # cached


def test_no_embedding_backend_means_keyword_only():
    """No embedding backend means keyword only."""
    cb._EMBEDDER_CACHE["none-test"] = None
    try:
        bank = _bank()
        bank.put(_hook("hook:a", ["attn"], code="attention"))
        assert bank.search("attention", embeddings="none-test")[0]["match"] == "keyword"
    finally:
        cb._EMBEDDER_CACHE.pop("none-test", None)


def test_learned_features_round_trip():
    """Learned features round trip."""
    bank = _bank()
    spec = {
        "id": "post_norm_only",
        "label": "Norms only after attention and MLP",
        "hooks": ["createTransformerDecoderBlock"],
        "when": [
            {"tensor": "layers.*.post_feedforward_layernorm.weight"},
            {"no_tensor": "layers.*.input_layernorm.weight"},
        ],
        "claims": {"tensors": ["layers.*.post_{attention,feedforward}_layernorm.weight"]},
    }
    bank.put_feature(spec, source_entry="hook:olmo2_block")
    got = bank.learned_features()
    assert got[0]["id"] == "post_norm_only" and got[0]["source"] == "learned"


# ----------------------------------------------------------- C++ reading
_CPP = r'''
#include "x.h"
namespace causallm {
/**
 * @brief doc
 */
Tensor Foo::createAttention(const int layer_id, int seq_len,
                            Tensor q) {
  // a { brace in a comment
  auto s = "a } brace in a string";
  if (x) { return q; }
  return Bar::createAttention(layer_id, seq_len, q);
}

Foo::Foo(json &cfg) : Bar(cfg), member{1} { setupParameters(cfg); }

void Foo::declared_only(int a);
} // namespace causallm
'''

_H = r'''
enum class Mode { A, B };
class WIN_EXPORT Foo : public Bar, virtual public Base {
public:
  Foo(json &cfg) : Bar(cfg) {}
  void setupParameters(json &cfg) override {
    Bar::setupParameters(cfg);
    helper();
  }
  void declared(int x);
  static constexpr const char *type = "foo_layer";
private:
  int v = 3;
};
'''


def test_out_of_class_definitions():
    """Out of class definitions."""
    defs = {d.method: d for d in cx.find_definitions(_CPP)}
    assert set(defs) == {"createAttention", "Foo"}
    create_attention = defs["createAttention"]
    assert create_attention.class_name == "Foo" and create_attention.signature.startswith(
        "Tensor Foo::createAttention("
    )
    assert (
        create_attention.body.strip().endswith("}")
        and "Bar::createAttention" in create_attention.body
    )
    assert "@brief doc" in create_attention.text and "@brief" not in create_attention.signature


def test_inline_methods_and_classes():
    """Inline methods and classes."""
    inline = {d.method: d for d in cx.find_inline_methods(_H)}
    assert set(inline) == {"Foo", "setupParameters"}
    assert inline["setupParameters"].signature.startswith("void setupParameters(")
    classes = {c.name: c.bases for c in cx.find_classes(_H)}
    assert classes == {"Foo": ["Bar", "Base"]}                  # the enum is not a class
    assert cx.find_layer_types(_H) == ["foo_layer"]


def test_calls_and_final_overriders():
    """Calls and final overriders."""
    qualified, plain = cx.calls_in(
        "{ Bar::setupParameters(c); helper(); obj.other(); p->far(); this->mine(); }"
    )
    assert qualified == [("Bar", "setupParameters")]
    assert "helper" in plain and "mine" in plain and "other" not in plain and "far" not in plain
    bases = {
        "XCausalLM": ["CausalLM", "XTransformer"],
        "CausalLM": ["Transformer"],
        "XTransformer": ["Transformer"],
    }
    defined = {"Transformer": {"createAttention", "createMlp"}, "XTransformer": {"createAttention"},
               "CausalLM": {"constructModel"}}
    finals = cx.final_overriders("XCausalLM", bases, defined)
    assert finals == {
        "createAttention": "XTransformer",
        "createMlp": "Transformer",
        "constructModel": "CausalLM",
    }
