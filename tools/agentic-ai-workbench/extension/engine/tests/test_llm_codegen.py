"""
Tests for the CausalLM LLM codegen pipeline.

The LLM is mocked throughout, which tests the surrounding machinery far more
rigorously than a live model would: every failure mode can be produced on
demand and asserted on, including ones a real model only hits occasionally.

Covers:
  * plan resolution against a real traced semantic_ir
  * skeleton emission (diamond ctor, exact signatures, deterministic
    registerCustomLayers)
  * body splicing through nested braces, comments, and string literals
  * validation catching the compile-clean failure modes
  * the correction/retry loop, including give-up behaviour
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.cpp import llm_codegen, model_plan, skeleton  # noqa: E402
from agents.cpp.plan_validator import validate  # noqa: E402

# A minimal traced IR shaped like the real thing: qwen3-style (q/k norm,
# gated silu MLP, no sandwich norms), so exactly one hook deviates.
_QWEN3_LIKE = {
    "architecture": "qwen3",
    "hidden_size": 1024,
    "vocab_size": 151936,
    "num_layers": 2,
    "tied_embeddings": True,
    "embedding_scale": 1.0,
    "embedding_name": "model.embed_tokens",
    "lm_head_name": "lm_head",
    "final_norm": {"norm_type": "rms_norm", "epsilon": 1e-6},
    "decoder_layers": [
        {
            "index": i,
            "input_norm": {"norm_type": "rms_norm", "epsilon": 1e-6},
            "post_attention_norm": {"norm_type": "rms_norm", "epsilon": 1e-6},
            "attn_post_norm": None,
            "mlp_post_norm": None,
            "attention": {
                "num_heads": 16, "num_kv_heads": 8, "head_dim": 128,
                "q_norm": {"norm_type": "rms_norm", "epsilon": 1e-6,
                           "feature_size": 128},
                "k_norm": {"norm_type": "rms_norm", "epsilon": 1e-6,
                           "feature_size": 128},
                "rope_theta": 1000000.0,
                "max_position_embeddings": 40960,
                "sliding_window": None,
                "causal": True,
                "q_proj": {"input_size": 1024, "output_size": 2048,
                           "bias": False},
                "k_proj": {"input_size": 1024, "output_size": 1024,
                           "bias": False},
                "v_proj": {"input_size": 1024, "output_size": 1024,
                           "bias": False},
                "o_proj": {"input_size": 2048, "output_size": 1024,
                           "bias": False},
            },
            "mlp": {"gated": True, "activation": "silu",
                    "up_proj": {"input_size": 1024, "output_size": 3072},
                    "gate_proj": {"input_size": 1024, "output_size": 3072},
                    "down_proj": {"input_size": 3072, "output_size": 1024}},
        }
        for i in range(2)
    ],
}


def _state(**over):
    s = {
        "model_name": "Qwen/Qwen3-0.6B",
        "architecture": "Qwen3ForCausalLM",
        "semantic_ir": _QWEN3_LIKE,
        "hf_config": {"vocab_size": 151936, "hidden_size": 1024},
        "out_dir": tempfile.mkdtemp(prefix="wb_test_"),
    }
    s.update(over)
    return s


# --------------------------------------------------------------------- plan
def test_plan_detects_only_qk_norm_deviation():
    plan = model_plan.build_plan(_state())
    hooks = {h["hook"] for h in plan["holes"]}
    # q/k norm needs createAttention; reshaped_rms_norm is unregistered so
    # registerCustomLayers is required. A gated silu MLP and plain block match
    # the base defaults exactly, so neither should appear.
    assert "createAttention" in hooks, hooks
    assert "registerCustomLayers" in hooks, hooks
    assert "createMlp" not in hooks, "gated silu MLP matches the base default"
    assert "createTransformerDecoderBlock" not in hooks, "no sandwich norms"
    assert plan["needs_registration"] == ["reshaped_rms_norm"]
    assert plan["class_name"] == "Qwen3CausalLM"
    assert plan["transformer_class"] == "Qwen3Transformer"
    assert plan["file_stem"] == "qwen3"


def test_plan_requires_semantic_ir():
    try:
        model_plan.build_plan({"architecture": "X"})
    except ValueError as exc:
        assert "semantic_ir" in str(exc)
    else:
        raise AssertionError("expected ValueError without semantic_ir")


# ----------------------------------------------------------------- skeleton
def test_skeleton_emits_virtual_base_diamond():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    h = sk["header"]
    # The most-derived class must initialise the virtual Transformer base
    # itself, first, with ModelType::CAUSALLM.
    assert "Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM)" in h
    assert "class Qwen3Transformer : virtual public Transformer" in h
    assert "class Qwen3CausalLM : public CausalLM, public Qwen3Transformer" in h
    assert "#ifndef __QWEN3_CAUSAL_LM_H__" in h


def test_skeleton_signature_is_exact():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    # The 7-arg base signature. Earlier generators emitted a 2-arg variant.
    assert "createAttention(const int layer_id, int seq_len" in sk["source"]
    assert "Tensor query, Tensor key" in sk["source"]


def test_registration_is_deterministic_not_a_hole():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    hooks = {h["hook"] for h in sk["holes"]}
    assert "registerCustomLayers" not in hooks, \
        "registration is boilerplate and must not be left to the model"
    # Both bases must be fanned out explicitly because of the diamond.
    assert "CausalLM::registerCustomLayers();" in sk["source"]
    assert "Qwen3Transformer::registerCustomLayers();" in sk["source"]
    assert "nntrainer::createLayer<causallm::ReshapedRMSNormLayer>" in sk["source"]


# ------------------------------------------------------------------ splice
_GOOD_BODY = '''```cpp
// FILE: qwen3_causallm.cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true")}));
  Tensor q = wq(query);
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor q_normed = q_norm(q);
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE)}));
  Tensor k = wk(key);
  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor k_normed = k_norm(k);
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE)}));
  Tensor v = wv(value);
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE)}));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM)}));
  return wo(a);
}
```'''


def test_splice_handles_nested_braces_and_literals():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    src, unfilled = llm_codegen._splice_bodies(
        sk["source"], _GOOD_BODY, plan, sk["holes"]
    )
    assert unfilled == [], unfilled
    assert "===FILL:" not in src, "all markers should be filled"
    # Our signature survives; the model's body is inside it.
    assert "createAttention(const int layer_id, int seq_len" in src
    assert "mha({q_normed, k_normed, v, cache_k, cache_v})" in src


def test_splice_reports_missing_bodies_and_keeps_markers():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    src, unfilled = llm_codegen._splice_bodies(
        sk["source"], "```cpp\n// nothing useful\n```", plan, sk["holes"]
    )
    assert unfilled == ["createAttention"], unfilled
    # A partial result is still a valid file, with the gap visible.
    assert "===FILL:1===" in src


# --------------------------------------------------------------- validation
def test_validator_accepts_good_spliced_output():
    plan = model_plan.build_plan(_state())
    sk = skeleton.emit_skeletons(plan)
    src, _ = llm_codegen._splice_bodies(sk["source"], _GOOD_BODY, plan,
                                        sk["holes"])
    r = validate(src, dict(plan, holes=sk["holes"]), strict=True)
    assert r["ok"], [f["detail"] for f in r["findings"]]


def test_validator_catches_compile_clean_bugs():
    plan = model_plan.build_plan(_state())
    bad = '''
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  LayerHandle wq(createLayer("fully_connected",
    {withKey("name", "model.layers.0.self_attn.q_proj")}));
  LayerHandle n(createLayer("rms_norm",
    {withKey("feature_size", "128"), withKey("weight_initializer", "ones")}));
  Tensor act = swiglu({up, gate});
  Tensor a = mha({q, k, v});
  auto [ck, cv] = createKVCachePlaceholders(layer_id, n_heads / GQA_SIZE);
  return applyRoPE(a);
}
'''
    r = validate(bad, dict(plan, holes=[]), strict=True)
    kinds = {f["kind"] for f in r["findings"]}
    # Every one of these compiles cleanly and fails silently at runtime.
    assert "hf-dotted-name" in kinds, kinds        # loads zero weights
    assert "rms-norm-feature-size" in kinds, kinds  # runtime throw
    assert "swiglu-remap" in kinds, kinds           # wrong numerics
    assert "mha-arity" in kinds, kinds              # cache never bound
    assert "kv-helper-arg" in kinds, kinds          # cache halved twice
    assert "forbidden-symbol" in kinds, kinds       # applyRoPE does not exist
    assert not r["ok"]


# ------------------------------------------------------------- retry loop
class _ScriptedLLM:
    """Returns a fixed sequence of responses, recording the prompts it saw."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.responses[min(len(self.prompts) - 1,
                                  len(self.responses) - 1)]


def _run_with(llm, **state_over):
    st = _state(cline_ums_token="x", **state_over)
    orig = llm_codegen._make_llm
    llm_codegen._make_llm = lambda s: llm
    try:
        return llm_codegen.generate(st), st
    finally:
        llm_codegen._make_llm = orig


def test_accepts_on_first_attempt_when_valid():
    llm = _ScriptedLLM(_GOOD_BODY)
    files, st = _run_with(llm)
    assert files is not None
    assert st["cpp_generation_method"] == "llm"
    assert len(llm.prompts) == 1, "should not retry a valid result"


def test_retries_with_correction_then_succeeds():
    broken = '''```cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  LayerHandle x(createLayer("rotary_embedding", {withKey("name", "x")}));
  return applyRoPE(x(query));
}
```'''
    llm = _ScriptedLLM(broken, _GOOD_BODY)
    files, st = _run_with(llm)
    assert files is not None
    assert st["cpp_generation_method"] == "llm"
    assert len(llm.prompts) == 2, "should have corrected once"
    # The correction prompt must carry the specific problems, not just say
    # "try again" -- that is what makes a weak model able to act on it.
    second = llm.prompts[1]
    assert "rotary_embedding" in second
    assert "GAP MARKER" in second


def test_retries_on_compile_error_then_succeeds():
    """
    Semantic validation passes but compilation fails, triggering a retry with
    compile errors fed to the correction prompt.
    """
    if not _can_compile():
        print("      (skipped: no g++ or nntrainer headers)")
        return
    # Semantically valid but syntactically broken (unmatched brace).
    broken = '''```cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  LayerHandle wq(createLayer("fully_connected", {withKey("name", "wq")}));
  Tensor q = wq(query)
  return q;
}  // unmatched brace
```'''
    llm = _ScriptedLLM(broken, _GOOD_BODY)
    files, st = _run_with(llm)
    assert files is not None
    # Compile check converted the syntax error to a finding, which triggered
    # a correction/retry, which succeeded.
    assert st["cpp_generation_method"] == "llm"
    assert len(llm.prompts) == 2, "should have retried due to compile error"
    # The second prompt should mention the compile failure WITH the exact line
    second = llm.prompts[1]
    assert ("compile-error" in second or "syntax" in second.lower() or
            "missing" in second.lower() or "semicolon" in second.lower()), \
        "correction prompt should mention the compile failure"
    # Verify the fix includes the actual source context, not just a generic message
    assert ("wq(query)" in second or "Line" in second), \
        "correction prompt should include the actual source line with the error"


def test_gives_up_but_still_emits_reviewable_output():
    broken = '''```cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  return applyRoPE(query);
}
```'''
    llm = _ScriptedLLM(broken)
    files, st = _run_with(llm, cpp_max_attempts=2)
    assert len(llm.prompts) == 2
    # Emitting nothing would waste the run; the scaffolding is ours and
    # correct, so a reviewable file plus recorded findings is more useful.
    assert files is not None
    assert st["cpp_generation_method"] == "llm_unvalidated"
    assert st["cpp_validation"]["counts"]["errors"] > 0


def test_no_llm_emits_skeleton_with_markers():
    # claude_cli_enabled=False is required alongside the popped env vars --
    # otherwise this test's result depends on whether the machine running it
    # happens to have the Claude CLI installed/authenticated (it does in
    # dev/CI sandboxes with Claude Code available), which would silently
    # turn "no LLM configured" into a real, non-deterministic LLM call.
    st = _state(claude_cli_enabled=False)
    os.environ.pop("CLINE_UMS_TOKEN", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    files = llm_codegen.generate(st)
    assert files is not None
    assert st["cpp_generation_method"] == "skeleton_unfilled"
    assert "===FILL:1===" in files.source
    # The header never has holes -- it is fully determined.
    assert "===FILL:" not in files.header


def test_matching_base_graph_needs_no_llm():
    """A plain llama-shaped model should need zero generation."""
    ir = json.loads(json.dumps(_QWEN3_LIKE))
    for l in ir["decoder_layers"]:            # drop q/k norm -> plain llama
        l["attention"]["q_norm"] = None
        l["attention"]["k_norm"] = None
    st = _state(semantic_ir=ir)
    plan = model_plan.build_plan(st)
    assert plan["holes"] == [], plan["holes"]
    files = llm_codegen.generate(st)
    assert files is not None
    assert st["cpp_generation_method"] == "skeleton_only"
    assert "===FILL:" not in files.source


# ------------------------------------------------------------ compile check
#
# These are the highest-value tests here. The virtual-inheritance
# "no unique final overrider" bug -- CausalLM and XTransformer both overriding
# setupParameters/constructModel, with neither dominating -- passed every
# structural assertion above and was caught only by invoking g++. Anything
# involving the diamond has to be verified by a real compiler.
_CAUSALLM = "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM"


def _can_compile() -> bool:
    import shutil
    return (
        shutil.which("g++") is not None
        and os.path.isfile(os.path.join(_CAUSALLM, "llm_util.hpp"))
        and os.path.isfile("/usr/local/include/nntrainer/layer.h")
    )


def _compile(gen_dir: str, source_path: str) -> tuple:
    """
    Syntax-only compile of a generated pair.

    `gen_dir` goes FIRST on the include path and the hand-written model dirs
    are omitted, so `#include <x_causallm.h>` can only resolve to the generated
    header. Without that, a same-named hand-written header shadows it and the
    check passes without ever reading the generated code.
    """
    import subprocess
    cmd = [
        "g++", "-std=c++17", "-fsyntax-only",
        f"-I{gen_dir}",
        "-I/usr/local/include/nntrainer",
        f"-I{_CAUSALLM}",
        f"-I{_CAUSALLM}/models",
        f"-I{_CAUSALLM}/layers",
        f"-I{_CAUSALLM}/third_party/json/include",
        source_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stderr


def _emit_to_disk(state_dict, raw_llm_output=None):
    plan = model_plan.build_plan(state_dict)
    sk = skeleton.emit_skeletons(plan)
    source = sk["source"]
    if raw_llm_output is not None:
        source, _ = llm_codegen._splice_bodies(source, raw_llm_output, plan,
                                               sk["holes"])
    d = tempfile.mkdtemp(prefix="wb_compile_")
    with open(os.path.join(d, sk["header_filename"]), "w") as f:
        f.write(sk["header"])
    src_path = os.path.join(d, sk["source_filename"])
    with open(src_path, "w") as f:
        f.write(source)
    return d, src_path


def test_compile_generated_pair_with_bodies():
    if not _can_compile():
        print("      (skipped: no g++ or nntrainer headers)")
        return
    d, src = _emit_to_disk(_state(), _GOOD_BODY)
    rc, err = _compile(d, src)
    assert rc == 0, f"generated code failed to compile:\n{err[:2500]}"


def test_compile_skeleton_with_unfilled_holes():
    """
    A skeleton whose holes were never filled must still compile.

    This is what makes a failed generation useful rather than wasted: the
    contributor gets a valid file with clearly marked TODOs instead of
    something that has to be repaired before it will even parse.
    """
    if not _can_compile():
        print("      (skipped: no g++ or nntrainer headers)")
        return
    real = "/storage_data/snap/Prachi/nntrainer_out/state.json"
    if not os.path.isfile(real):
        print("      (skipped: no traced state.json)")
        return
    with open(real) as f:
        st = json.load(f)
    st["out_dir"] = tempfile.mkdtemp(prefix="wb_")
    d, src = _emit_to_disk(st)          # gemma3: 5 holes, all unfilled
    rc, err = _compile(d, src)
    assert rc == 0, f"unfilled skeleton failed to compile:\n{err[:2500]}"


def test_compile_detects_a_deliberately_broken_header():
    """
    Control for the two tests above.

    If the generated header were being shadowed by a hand-written one of the
    same name, those tests would pass without compiling any generated code at
    all. Breaking the generated header must therefore break the compile.
    """
    if not _can_compile():
        print("      (skipped: no g++ or nntrainer headers)")
        return
    d, src = _emit_to_disk(_state(), _GOOD_BODY)
    hdr = os.path.join(d, "qwen3_causallm.h")
    with open(hdr) as f:
        text = f.read()
    with open(hdr, "w") as f:
        f.write(text.replace(
            "class Qwen3Transformer : virtual public Transformer {",
            "class Qwen3Transformer : virtual public Transformer { CANARY",
        ))
    rc, err = _compile(d, src)
    assert rc != 0, "the generated header is being shadowed -- compile " \
                    "tests above are not testing generated code"
    assert "CANARY" in err


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
