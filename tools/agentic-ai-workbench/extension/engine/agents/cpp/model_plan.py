"""
ModelPlan: the deterministic resolver that sits in front of the LLM.

The core idea
-------------
`causallm::Transformer` / `causallm::CausalLM` have ZERO pure virtual methods.
They already build the complete llama-shaped graph:

    input0 -> embedding0 -> N x [attn_norm -> attention -> add
                                -> ffn_norm -> mlp -> add] -> output_norm
                                                           -> lm head

So generating a model file is not "write a model". It is "name the handful of
places this architecture deviates from that default, and write only those".

This module makes that comparison in Python, deterministically. It reads the
traced `semantic_ir` and produces a ModelPlan naming exactly which hooks need
overriding and with which resolved values. The LLM then transcribes.

Why this matters for weak models
--------------------------------
Every decision made here is a decision a small model cannot get wrong. It does
no arithmetic (head_dim, gqa, ffn width are all resolved), makes no
architectural judgement ("does this model tie embeddings?"), and picks no layer
types. Its remaining job -- render a resolved plan into C++ using a closed menu
-- is a transformation task, which small models do well.

Input priority (established by inspecting a real 1.3 MB state.json):
  1. state["semantic_ir"]        -- richest: real per-projection sizes, all four
                                    norm slots, q/k norm, gated+activation,
                                    embedding_scale, tied_embeddings
  2. state["hf_config"]          -- fields semantic_ir drops: per-layer
                                    `layer_types`, nested `rope_parameters`,
                                    softcapping, query_pre_attn_scalar
  3. state["nntrainer_graph_ir"] -- authoritative target types/attrs, but 345
                                    nodes for 18 layers and null on every
                                    shape/dtype field, so used only for
                                    metadata flags
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from knowledge.causallm_kb import LAYER_CATALOG, unregistered_types

# Activations the base MLP path handles natively. `Transformer::createMlp`
# builds up+gate -> swiglu -> down, which is only correct for SiLU/swish
# gating; anything else needs an override.
_SWIGLU_ACTIVATIONS = frozenset({"silu", "swish"})

# HF activation name -> nntrainer activation name. Mirrors
# api/lowering/nntrainer/nntrainer_names.py. Deliberately raises on unknown
# input rather than guessing -- an invented activation string reaches the C++
# as a silently-wrong numeric path.
_ACTIVATION_MAP: Dict[str, str] = {
    "silu": "swish", "swish": "swish",
    "gelu": "gelu",
    "gelu_pytorch_tanh": "tanh_gelu",
    "gelu_new": "tanh_gelu",
    "relu": "relu", "tanh": "tanh", "sigmoid": "sigmoid",
    "mish": "mish", "elu": "elu", "selu": "selu", "softmax": "softmax",
}

# Values that must appear in the C++ as bare expressions, not string literals.
# The lowerer models these as RawCppExpr (a str subclass), which is
# indistinguishable from a plain string once it round-trips through JSON -- so
# the plan tags them explicitly and the prompt renders them unquoted.
RAW_CPP_MARKER = "@raw:"


def _raw(expr: str) -> str:
    """Tag a value as a literal C++ expression rather than a quoted string."""
    return f"{RAW_CPP_MARKER}{expr}"


def is_raw(value: object) -> bool:
    return isinstance(value, str) and value.startswith(RAW_CPP_MARKER)


def raw_expr(value: str) -> str:
    return value[len(RAW_CPP_MARKER):]


# ---------------------------------------------------------------------------
# Naming. Transcribed from the shipping models, NOT from manifest.py.
#
# manifest.py's _TAG_TARGET_NAMES disagrees with the shipping C++ on the MLP
# projections (it says gate_proj/up_proj/down_proj; transformer.cpp emits
# layer<i>_ffn_gate/_ffn_up/_ffn_down). The C++ is what weight loading actually
# matches against at runtime, so the C++ wins here.
# ---------------------------------------------------------------------------
def layer_name(tag: str, layer_id_expr: str = 'std::to_string(layer_id)') -> str:
    """Build the canonical per-layer name expression for `tag`."""
    return f'"layer" + {layer_id_expr} + "_{tag}"'


CANONICAL_TAGS: Dict[str, str] = {
    "attn_norm": "attention_norm",
    "wq": "wq", "wk": "wk", "wv": "wv", "wo": "attention_out",
    "q_norm": "q_norm", "k_norm": "k_norm",
    "attention": "attention",
    "decoder_add": "decoder_add",
    "ffn_norm": "ffn_norm",
    "ffn_up": "ffn_up", "ffn_gate": "ffn_gate",
    "ffn_swiglu": "ffn_swiglu", "ffn_down": "ffn_down",
    "decoder_output": "decoder_output",
    "attn_post_norm": "attn_post_norm",
    "ffn_post_norm": "ffn_post_norm",
}

GLOBAL_NAMES: Dict[str, str] = {
    "input": "input0",
    "embedding": "embedding0",
    "final_norm": "output_norm",
    "lm_head": "output_of_causallm",
}


# ---------------------------------------------------------------------------
# Class naming
# ---------------------------------------------------------------------------
def _class_names(architecture: str, arch_key: str) -> Tuple[str, str, str]:
    """
    Return (transformer_class, causallm_class, file_stem).

    Generated classes intentionally never share a symbol with a handwritten
    model.  A generated Qwen3 is therefore Qwen3WorkbenchCausalLM, not
    Qwen3CausalLM.  This lets a Meson option select the generated target
    without ODR/link conflicts with the shipping implementation.
    """
    stem = re.sub(r"[^a-z0-9_]", "", (arch_key or "model").lower())
    base = re.sub(r"ForCausalLM$|ForConditionalGeneration$|Model$", "",
                  architecture or "")
    base = re.sub(r"[^A-Za-z0-9]", "", base) or stem.title()
    return (f"{base}WorkbenchTransformer", f"{base}WorkbenchCausalLM",
            f"{stem}_workbench")


# ---------------------------------------------------------------------------
# Delta detection. One predicate per overridable hook.
#
# Each returns (needs_override, reason). Keeping them separate and named makes
# the plan self-explaining: the prompt shows the contributor *why* a hook is
# being overridden, which is also what makes a wrong plan reviewable.
# ---------------------------------------------------------------------------
def _attention_delta(attn: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    if attn.get("q_norm") or attn.get("k_norm"):
        reasons.append("per-head q/k norm (base createAttention has none)")
    if attn.get("use_sink"):
        reasons.append("attention sinks")
    rope_type = (attn.get("rope_scaling_type") or "default").lower()
    if rope_type not in ("default", "", "none"):
        reasons.append(f"non-default rope scaling ({rope_type})")
    if attn.get("q_proj", {}).get("bias") or attn.get("k_proj", {}).get("bias"):
        reasons.append("biased qkv projections")
    return bool(reasons), reasons


def _mlp_delta(mlp: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    if not mlp.get("gated", True):
        reasons.append("MLP is not gated (base assumes gated SwiGLU)")
    act = (mlp.get("activation") or "silu").lower()
    if act not in _SWIGLU_ACTIVATIONS:
        reasons.append(f"activation is {act}, not silu/swish, so swiglu "
                       f"does not apply")
    if mlp.get("num_experts"):
        reasons.append("mixture-of-experts MLP")
    return bool(reasons), reasons


def _block_delta(layer: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    if layer.get("attn_post_norm"):
        reasons.append("post-attention sandwich norm")
    if layer.get("mlp_post_norm"):
        reasons.append("post-MLP sandwich norm")
    return bool(reasons), reasons


def _construct_delta(ir: Dict, hf: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    scale = ir.get("embedding_scale")
    if scale not in (None, 0, 1, 1.0):
        reasons.append(f"embedding output is scaled by {scale}")
    if hf.get("final_logit_softcapping"):
        reasons.append("final logit softcapping")
    return bool(reasons), reasons


def _setup_delta(hf: Dict) -> Tuple[bool, List[str]]:
    """Config fields Transformer::setupParameters does not read."""
    extra = [
        k for k in (
            "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "query_pre_attn_scalar", "final_logit_softcapping",
            "attn_logit_softcapping", "layer_types",
            "sliding_window_pattern",  # Gemma3-specific: determines layer_types
        )
        if hf.get(k) is not None
    ]
    if extra:
        return True, [f"config fields not read by the base: {', '.join(extra)}"]
    return False, []


# ---------------------------------------------------------------------------
# Attention resolution
# ---------------------------------------------------------------------------
def _resolve_attention(attn: Dict, norm_eps: float) -> Dict:
    """
    Resolve the attention subgraph into an ordered layer list with every
    property value already decided.

    Sizing uses base-class members (`head_dim * n_heads`, `GQA_SIZE`, `DIM`)
    rather than baked literals, matching the shipping models -- a file with
    hardcoded dims silently breaks when nntr_config changes.
    """
    has_qk_norm = bool(attn.get("q_norm") or attn.get("k_norm"))
    layers: List[Dict] = [
        {
            "tag": "wq", "type": "fully_connected",
            "name": layer_name(CANONICAL_TAGS["wq"]),
            "props": {"unit": _raw("head_dim * n_heads"),
                      "disable_bias": "true",
                      "weight_initializer": "ones"},
            "input": "query", "output": "q",
        },
    ]
    if attn.get("q_norm"):
        layers.append({
            "tag": "q_norm", "type": "reshaped_rms_norm",
            "name": layer_name(CANONICAL_TAGS["q_norm"]),
            "props": {"packed": "false",
                      "epsilon": _raw("std::to_string(NORM_EPS)"),
                      "feature_size": _raw("std::to_string(head_dim)")},
            "input": "q", "output": "q_normed",
        })
    layers.append({
        "tag": "wk", "type": "fully_connected",
        "name": layer_name(CANONICAL_TAGS["wk"]),
        "props": {"unit": _raw("head_dim * n_heads / GQA_SIZE"),
                  "disable_bias": "true", "weight_initializer": "ones"},
        "input": "key", "output": "k",
    })
    if attn.get("k_norm"):
        layers.append({
            "tag": "k_norm", "type": "reshaped_rms_norm",
            "name": layer_name(CANONICAL_TAGS["k_norm"]),
            "props": {"packed": "false",
                      "epsilon": _raw("std::to_string(NORM_EPS)"),
                      "feature_size": _raw("std::to_string(head_dim)")},
            "input": "k", "output": "k_normed",
        })
    layers.append({
        "tag": "wv", "type": "fully_connected",
        "name": layer_name(CANONICAL_TAGS["wv"]),
        "props": {"unit": _raw("head_dim * n_heads / GQA_SIZE"),
                  "disable_bias": "true", "weight_initializer": "ones"},
        "input": "value", "output": "v",
    })

    # KV cache placeholders come from the base helper. Note it takes the FULL
    # head count -- it applies the GQA reduction internally. Passing the
    # already-reduced count is a real bug seen in earlier generated output.
    layers.append({
        "tag": "kv_cache", "type": "@helper",
        "call": "auto [cache_k, cache_v] = "
                "createKVCachePlaceholders(layer_id, n_heads);",
        "note": "pass n_heads, NOT n_heads / GQA_SIZE -- the helper reduces "
                "internally",
    })

    mha_props: Dict[str, object] = {
        "num_heads": _raw("n_heads"),
        "num_heads_kv": _raw("n_heads / GQA_SIZE"),
        "max_timestep": _raw("std::to_string(MAX_SEQ_LEN)"),
        "sliding_window": _raw("SLIDING_WINDOW"),
        "rope_theta": _raw("ROPE_THETA"),
        "max_position_embeddings": _raw("MAX_POSITION_EMBEDDINGS"),
        "max_new_tokens": _raw("std::to_string(NUM_TO_GENERATE)"),
        "is_causal": _raw('IS_CAUSAL ? "true" : "false"'),
    }
    if attn.get("use_sink"):
        mha_props["use_sink"] = "true"
    rope_type = (attn.get("rope_scaling_type") or "default").lower()
    if rope_type == "yarn":
        mha_props["rope_scaling_type"] = "yarn"
        mha_props["rope_scaling_factor"] = str(
            attn.get("rope_scaling_factor", 1.0))
        mha_props["rope_scaling_max_position_embeddings"] = str(
            attn.get("rope_scaling_max_position_embeddings", 4096))

    q_in = "q_normed" if attn.get("q_norm") else "q"
    k_in = "k_normed" if attn.get("k_norm") else "k"
    layers.append({
        "tag": "attention", "type": "mha_core",
        "name": layer_name(CANONICAL_TAGS["attention"]),
        "props": mha_props,
        "inputs": [q_in, k_in, "v", "cache_k", "cache_v"],
        "output": "a",
        "note": "input order is fixed: {q, k, v, cache_k, cache_v}",
    })
    layers.append({
        "tag": "wo", "type": "fully_connected",
        "name": layer_name(CANONICAL_TAGS["wo"]),
        "props": {"unit": _raw("DIM"), "disable_bias": "true",
                  "weight_initializer": "ones"},
        "input": "a", "output": "@return",
    })

    return {
        "has_qk_norm": has_qk_norm,
        "num_heads": attn.get("num_heads"),
        "num_kv_heads": attn.get("num_kv_heads"),
        "head_dim": attn.get("head_dim"),
        "layers": layers,
    }


def _resolve_mlp(mlp: Dict) -> Dict:
    """Resolve the MLP subgraph. Handles gated and ungated variants."""
    act_raw = (mlp.get("activation") or "silu").lower()
    act = _ACTIVATION_MAP.get(act_raw)
    if act is None:
        raise ValueError(
            f"activation '{act_raw}' has no nntrainer equivalent. Add it to "
            f"_ACTIVATION_MAP and to nntrainer_names.py, or the generated code "
            f"will use a silently-wrong activation."
        )

    if not mlp.get("gated", True):
        return {
            "gated": False, "activation": act,
            "layers": [
                {"tag": "ffn_up", "type": "fully_connected",
                 "name": layer_name(CANONICAL_TAGS["ffn_up"]),
                 "props": {"unit": _raw("hidden_dim"),
                           "disable_bias": "true"},
                 "input": "input", "output": "up"},
                {"tag": "act", "type": "activation",
                 "name": layer_name("ffn_act"),
                 "props": {"activation": act},
                 "input": "up", "output": "act_out"},
                {"tag": "ffn_down", "type": "fully_connected",
                 "name": layer_name(CANONICAL_TAGS["ffn_down"]),
                 "props": {"unit": _raw("dim"), "disable_bias": "true"},
                 "input": "act_out", "output": "@return"},
            ],
        }

    # Gated. If the activation is silu/swish the base class already does this
    # exactly -- no override needed, and `_mlp_delta` will have said so.
    if act == "swish":
        return {
            "gated": True, "activation": act, "matches_base": True,
            "layers": [
                {"tag": "ffn_up", "type": "fully_connected",
                 "name": layer_name(CANONICAL_TAGS["ffn_up"]),
                 "props": {"unit": _raw("hidden_dim"),
                           "disable_bias": "true"},
                 "input": "input", "output": "up"},
                {"tag": "ffn_gate", "type": "fully_connected",
                 "name": layer_name(CANONICAL_TAGS["ffn_gate"]),
                 "props": {"unit": _raw("hidden_dim"),
                           "disable_bias": "true"},
                 "input": "input", "output": "gate"},
                {"tag": "ffn_swiglu", "type": "swiglu",
                 "name": layer_name(CANONICAL_TAGS["ffn_swiglu"]),
                 "props": {},
                 "call": "Tensor act = swiglu({up, gate}, {1, 0});",
                 "output": "act",
                 "note": "the {1, 0} index remap is mandatory -- layers are "
                         "created up-then-gate to match nntrainer weight "
                         "order, and the remap corrects the wiring"},
                {"tag": "ffn_down", "type": "fully_connected",
                 "name": layer_name(CANONICAL_TAGS["ffn_down"]),
                 "props": {"unit": _raw("dim"), "disable_bias": "true"},
                 "input": "act", "output": "@return"},
            ],
        }

    # Gated with a non-swish activation (e.g. Gemma's tanh_gelu): swiglu is
    # wrong, so the gate is activated explicitly and multiplied.
    return {
        "gated": True, "activation": act, "matches_base": False,
        "layers": [
            {"tag": "ffn_up", "type": "fully_connected",
             "name": layer_name(CANONICAL_TAGS["ffn_up"]),
             "props": {"unit": _raw("hidden_dim"), "disable_bias": "true"},
             "input": "input", "output": "up"},
            {"tag": "ffn_gate", "type": "fully_connected",
             "name": layer_name(CANONICAL_TAGS["ffn_gate"]),
             "props": {"unit": _raw("hidden_dim"), "disable_bias": "true"},
             "input": "input", "output": "gate"},
            {"tag": "act", "type": "activation",
             "name": layer_name("ffn_act"),
             "props": {"activation": act},
             "input": "gate", "output": "gate_act"},
            {"tag": "mul", "type": "multiply",
             "name": layer_name("ffn_mul"),
             "props": {}, "inputs": ["gate_act", "up"], "output": "act"},
            {"tag": "ffn_down", "type": "fully_connected",
             "name": layer_name(CANONICAL_TAGS["ffn_down"]),
             "props": {"unit": _raw("dim"), "disable_bias": "true"},
             "input": "act", "output": "@return"},
        ],
    }


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------
def _collect_gaps(resolved: Dict) -> List[Dict]:
    """
    Find layers the plan wants that no CausalLM layer implements.

    Returns structured entries rather than prose, so the UI can render them as
    contributor tasks and a later auto-fix pass can consume them without
    re-parsing generated C++.
    """
    from knowledge.causallm_kb import all_known_types

    known = all_known_types()
    gaps: List[Dict] = []
    for section, body in resolved.items():
        if not isinstance(body, dict):
            continue
        for layer in body.get("layers", []):
            t = layer.get("type", "")
            if t.startswith("@") or t in known:
                continue
            gaps.append({
                "op": layer.get("tag", t),
                "suggested_type": t,
                "section": section,
                "inputs": len(layer.get("inputs", [layer.get("input")])),
                "outputs": 1,
                "after": layer.get("input", ""),
                "reason": f"no layer in Applications/CausalLM/layers/ "
                          f"registers type '{t}'",
                "to_contribute": [
                    f"add layers/{t}_layer.h and .cpp",
                    f'give it: static constexpr const char *type = "{t}";',
                    "add it to layers/meson.build",
                    "registerFactory it in registerCustomLayers()",
                ],
            })
    return gaps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_plan(state: Dict) -> Dict:
    """
    Project state["semantic_ir"] into a ModelPlan for the prompt.

    The plan is authoritative: every layer type, property value, and layer name
    is decided here so the LLM never has to. Raises ValueError if semantic_ir
    is absent -- that means no architecture adapter matched, and generating
    from raw hf_config would be guesswork.
    """
    ir = state.get("semantic_ir")
    if not ir:
        raise ValueError(
            "no semantic_ir in state -- no architecture adapter matched this "
            "model. Add an adapter under api/semantic/ before generating; "
            "guessing the graph from hf_config alone produces unverifiable code."
        )

    hf = state.get("hf_config") or {}
    architecture = state.get("architecture") or ""
    arch_key = ir.get("architecture") or ""
    transformer_cls, causallm_cls, stem = _class_names(architecture, arch_key)

    layers = ir.get("decoder_layers") or []
    if not layers:
        raise ValueError("semantic_ir has no decoder_layers")
    layer0 = layers[0]
    attn = layer0.get("attention") or {}
    mlp = layer0.get("mlp") or {}
    norm_eps = (layer0.get("input_norm") or {}).get("epsilon", 1e-5)

    # --- which hooks deviate from the base class defaults ------------------
    holes: List[Dict] = []
    resolved: Dict[str, object] = {
        "architecture": arch_key,
        "hidden_size": ir.get("hidden_size"),
        "vocab_size": ir.get("vocab_size"),
        "num_layers": ir.get("num_layers"),
        "tied_embeddings": ir.get("tied_embeddings"),
        "embedding_scale": ir.get("embedding_scale"),
        "norm_eps": norm_eps,
        "uniform_layers": _uniform(layers),
        "global_names": GLOBAL_NAMES,
    }

    need_attn, why_attn = _attention_delta(attn)
    if need_attn:
        resolved["attention"] = _resolve_attention(attn, norm_eps)
        holes.append({
            "id": len(holes) + 1,
            "hook": "createAttention",
            "goal": "; ".join(why_attn),
            "layers": [l["type"] for l in resolved["attention"]["layers"]
                       if not l["type"].startswith("@")],
        })

    need_mlp, why_mlp = _mlp_delta(mlp)
    if need_mlp:
        resolved["mlp"] = _resolve_mlp(mlp)
        holes.append({
            "id": len(holes) + 1,
            "hook": "createMlp",
            "goal": "; ".join(why_mlp),
            "layers": [l["type"] for l in resolved["mlp"]["layers"]
                       if not l["type"].startswith("@")],
        })

    need_block, why_block = _block_delta(layer0)
    if need_block:
        resolved["block"] = {
            "attn_post_norm": bool(layer0.get("attn_post_norm")),
            "mlp_post_norm": bool(layer0.get("mlp_post_norm")),
            "norm_eps": norm_eps,
        }
        holes.append({
            "id": len(holes) + 1,
            "hook": "createTransformerDecoderBlock",
            "goal": "; ".join(why_block),
            "layers": ["rms_norm", "addition"],
        })

    need_construct, why_construct = _construct_delta(ir, hf)
    if need_construct:
        resolved["construct"] = {
            "embedding_scale": ir.get("embedding_scale"),
            "final_logit_softcapping": hf.get("final_logit_softcapping"),
            "tied_embeddings": ir.get("tied_embeddings"),
        }
        holes.append({
            "id": len(holes) + 1,
            "hook": "constructModel",
            "goal": "; ".join(why_construct),
            "layers": ["embedding_layer", "scalar_multiply", "rms_norm"],
        })

    need_setup, why_setup = _setup_delta(hf)
    if need_setup:
        resolved["setup_extra"] = {
            k: hf[k] for k in (
                "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "query_pre_attn_scalar", "final_logit_softcapping",
                "attn_logit_softcapping",
            ) if hf.get(k) is not None
        }
        holes.append({
            "id": len(holes) + 1,
            "hook": "setupParameters",
            "goal": "; ".join(why_setup),
            "layers": [],
        })

    # --- layer types actually used, and which need registering -------------
    layer_types = sorted({
        l["type"]
        for section in resolved.values() if isinstance(section, dict)
        for l in section.get("layers", [])
        if isinstance(l, dict) and not l.get("type", "@").startswith("@")
    })
    needs_registration = [t for t in layer_types if t in unregistered_types()]
    if needs_registration:
        holes.append({
            "id": len(holes) + 1,
            "hook": "registerCustomLayers",
            "goal": f"register {', '.join(needs_registration)} -- not "
                    f"registered by the base class, so createLayer() would "
                    f"throw at runtime",
            "layers": needs_registration,
        })

    gaps = _collect_gaps(resolved)

    return {
        "model_id": state.get("model_name", ""),
        "architecture": architecture,
        "arch_key": arch_key,
        "class_name": causallm_cls,
        "transformer_class": transformer_cls,
        "base_class": "causallm::CausalLM",
        "file_stem": stem,
        "holes": holes,
        "layer_types": layer_types,
        "needs_registration": needs_registration,
        "gaps": gaps,
        "resolved": resolved,
    }


def _uniform(layers: List[Dict]) -> bool:
    """
    True when every decoder layer has the same structure.

    Non-uniform models (Gemma's sliding/full alternation) cannot be emitted as
    a single templated loop, which is exactly the condition that made the old
    template emitter raise and fall back to copying a hand-written model.
    """
    if len(layers) <= 1:
        return True

    def sig(l: Dict) -> tuple:
        a = l.get("attention") or {}
        m = l.get("mlp") or {}
        return (
            bool(a.get("q_norm")), bool(a.get("k_norm")),
            a.get("num_heads"), a.get("num_kv_heads"), a.get("head_dim"),
            a.get("sliding_window"), a.get("rope_theta"),
            m.get("gated"), m.get("activation"),
            bool(l.get("attn_post_norm")), bool(l.get("mlp_post_norm")),
        )

    first = sig(layers[0])
    return all(sig(l) == first for l in layers[1:])


def plan_summary(plan: Dict) -> str:
    """One-paragraph human summary, for the workbench log and chat panel."""
    holes = plan.get("holes", [])
    if not holes:
        return (
            f"{plan['architecture']} matches the base llama-shaped graph "
            f"exactly. No overrides needed -- the generated file only "
            f"declares the class pair and inherits everything."
        )
    parts = ", ".join(f"{h['hook']} ({h['goal']})" for h in holes)
    gap_note = ""
    if plan.get("gaps"):
        gap_note = (f" {len(plan['gaps'])} unmapped op(s) will be emitted as "
                    f"marked gaps for a contributor to fill.")
    return (f"{plan['architecture']} deviates from the base graph in "
            f"{len(holes)} place(s): {parts}.{gap_note}")
