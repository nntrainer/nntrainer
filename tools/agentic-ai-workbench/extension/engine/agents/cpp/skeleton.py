"""
Deterministic skeleton emitter for CausalLM model files.

No LLM involved. Everything here is either fixed boilerplate or mechanically
derived from the ModelPlan, and it deliberately covers the parts of the file
that are hardest to get right by generation:

  * the virtual-base constructor diamond -- `XCausalLM` must initialise
    `Transformer` itself, first, with `ModelType::CAUSALLM`, because
    `Transformer` is a *virtual* base of both `CausalLM` and `XTransformer`.
    Getting this wrong is a compile error with an opaque message, and it is
    not guessable from the surrounding code.
  * exact method signatures -- earlier generated output had
    `createAttention(const int layer_id, Tensor hidden_states)` against a real
    base signature of
    `createAttention(const int, int, int, int, Tensor, Tensor, Tensor)`.
    Emitting the signature ourselves makes that class of failure impossible.
  * the `registerCustomLayers` fan-out -- the diamond means `XCausalLM` must
    call BOTH `CausalLM::registerCustomLayers()` and
    `XTransformer::registerCustomLayers()`; neither happens automatically.
  * include guards, SPDX banner, namespace, destructors, and the
    `architectures` convention member.

What is left to the LLM is only the body of each planned hook, between
`// ===FILL:n===` and `// ===END:n===` markers. That is a transcription task
over a resolved plan, which is what small models can actually do.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from knowledge.causallm_kb import LAYER_CATALOG

# Exact base-class signatures, transcribed from
# Applications/CausalLM/models/transformer.h and causal_lm.h.
# Keys match ModelPlan hole `hook` values.
_SIGNATURES: Dict[str, Dict[str, str]] = {
    "createAttention": {
        "ret": "Tensor",
        "name": "createAttention",
        # Parameter groups, one per source line. Continuation lines are aligned
        # under the opening paren at emit time, since the column depends on the
        # class name length and hand-written alignment would drift.
        "params": ["const int layer_id, int seq_len", "int n_heads, int head_dim",
                   "Tensor query, Tensor key", "Tensor value"],
        "on": "transformer",
    },
    "createMlp": {
        "ret": "Tensor",
        "name": "createMlp",
        "params": ["const int layer_id, int dim, int hidden_dim", "Tensor input"],
        "on": "transformer",
    },
    "createTransformerDecoderBlock": {
        "ret": "Tensor",
        "name": "createTransformerDecoderBlock",
        "params": ["const int layer_id", "Tensor input"],
        "on": "transformer",
    },
    "constructModel": {
        "ret": "std::pair<Tensor, Tensor>",
        "name": "constructModel",
        "params": [],
        "on": "transformer",
    },
    "setupParameters": {
        "ret": "void",
        "name": "setupParameters",
        "params": ["json &cfg, json &generation_cfg", "json &nntr_cfg"],
        "on": "transformer",
    },
    "registerCustomLayers": {
        "ret": "void",
        "name": "registerCustomLayers",
        "params": [],
        "on": "both",
    },
}


# Hooks that causallm::CausalLM ALSO overrides.
#
# This is the subtlety that makes the diamond bite. If both CausalLM and
# XTransformer override the same Transformer virtual, then in XCausalLM neither
# override dominates the other and C++ reports "no unique final overrider" --
# a hard error, at the class definition, with a message that gives no hint
# about the fix. The most-derived class must re-declare the member to
# disambiguate, and explicitly fan out to the bases it wants.
#
# Verified against the shipping models: gemma3_causallm.h does exactly this for
# setupParameters, and gemma4_causallm.h/.cpp for constructModel.
_CAUSALLM_ALSO_OVERRIDES = frozenset({
    "setupParameters", "constructModel", "registerCustomLayers",
})

# The LM-head append from CausalLM::constructModel (causal_lm.cpp:223-245).
#
# When XTransformer overrides constructModel, XCausalLM's disambiguating
# override must call XTransformer::constructModel() and then re-do this head
# append -- delegating to CausalLM::constructModel() instead would run the
# *base* Transformer body and discard the architecture's custom graph. It is
# fixed boilerplate driven only by TIE_WORD_EMBEDDINGS, so it is emitted here
# rather than asked of a model.
_LM_HEAD_APPEND = """\
  // Base graph from the architecture-specific override above, then the LM
  // head. Note this calls {tcls}::constructModel(), NOT
  // CausalLM::constructModel() -- the latter would rebuild the plain base
  // graph and discard this architecture's overrides.
  auto [x, h] = {tcls}::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {{
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  }};

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {{x, y}};
"""


def _declaration(sig: Dict) -> str:
    """In-class declaration, e.g. `Tensor createAttention(...) override;`."""
    head = f'{sig["ret"]} {sig["name"]}('
    if not sig["params"]:
        return f"{head}) override;"
    pad = " " * (len(head) + 2)  # +2 for the two-space member indent
    joined = (",\n" + pad).join(sig["params"])
    return f"{head}{joined}) override;"


def _definition(sig: Dict, cls: str) -> str:
    """Out-of-class definition opener, e.g. `Tensor Foo::createAttention(...) {`."""
    head = f'{sig["ret"]} {cls}::{sig["name"]}('
    if not sig["params"]:
        return f"{head}) {{"
    pad = " " * len(head)
    joined = (",\n" + pad).join(sig["params"])
    return f"{head}{joined}) {{"

_SPDX_H = """\
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   {filename}
 * @date   {date}
 * @brief  {arch} causal language model for the nntrainer CausalLM application
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Generated by the nntrainer agentic AI workbench
 * @bug    No known bugs except for NYI items
 *
 * AUTO-GENERATED from {model_id}.
 *
 * Overrides in this file, and why each is needed:
{override_notes}
 *
 * Everything not overridden is inherited from causallm::Transformer /
 * causallm::CausalLM, which already build the standard decoder graph.
 */
"""


def _override_notes(plan: Dict) -> str:
    lines = []
    for h in plan.get("holes", []):
        lines.append(f" *   - {h['hook']}: {h['goal']}")
    if not lines:
        lines.append(" *   (none -- this architecture matches the base graph)")
    return "\n".join(lines)


def _custom_headers(plan: Dict) -> List[str]:
    """Headers for catalog layers this model uses that need including."""
    out = []
    for t in plan.get("layer_types", []):
        entry = LAYER_CATALOG.get(t)
        if entry and entry.get("header"):
            out.append(entry["header"])
    return sorted(set(out))


def _registration_body(plan: Dict, transformer_cls: str) -> str:
    """
    The full registerCustomLayers body -- emitted deterministically rather
    than left as a hole.

    It is pure boilerplate derived entirely from `plan["needs_registration"]`,
    with a fixed try/catch shape, so there is nothing for a model to decide and
    no reason to spend a hole on it. Getting it wrong means a runtime throw, so
    it is also the last thing worth risking to generation.
    """
    need = plan.get("needs_registration", [])
    if not need:
        return "  // No unregistered custom layers -- the base class already\n" \
               "  // registers everything this model uses.\n"
    regs = []
    for t in need:
        cls = LAYER_CATALOG[t]["cls"]
        regs.append(
            f"  try {{\n"
            f"    app_context->registerFactory(\n"
            f"      nntrainer::createLayer<{cls}>);\n"
            f"  }} catch (std::invalid_argument &e) {{\n"
            f"    // Already registered -- expected when several models that\n"
            f"    // share this layer are constructed in one process.\n"
            f"    std::cerr << \"failed to register factory, reason: \"\n"
            f"              << e.what() << std::endl;\n"
            f"  }}"
        )
    return (
        "  auto &ct_engine = nntrainer::Engine::Global();\n"
        "  auto app_context = static_cast<nntrainer::AppContext *>(\n"
        "    ct_engine.getRegisteredContext(\"cpu\"));\n\n"
        + "\n".join(regs) + "\n"
    )


def emit_header(plan: Dict, date: str = "2025") -> str:
    """Emit the complete .h file. Contains no FILL markers -- fully determined."""
    tcls = plan["transformer_class"]
    ccls = plan["class_name"]
    stem = plan["file_stem"]
    guard = f"__{stem.upper()}_CAUSAL_LM_H__"
    arch = plan.get("architecture", "")

    hooks = [h["hook"] for h in plan.get("holes", [])]
    # registerCustomLayers is always declared when anything needs registering,
    # even if the resolver did not raise it as a hole.
    if plan.get("needs_registration") and "registerCustomLayers" not in hooks:
        hooks.append("registerCustomLayers")

    t_decls, c_decls = [], []
    for hook in hooks:
        sig = _SIGNATURES.get(hook)
        if not sig:
            continue
        t_decls.append("  " + _declaration(sig))

        if hook not in _CAUSALLM_ALSO_OVERRIDES:
            continue

        # CausalLM overrides this virtual too, so XCausalLM must re-declare it
        # to provide a unique final overrider. setupParameters gets the inline
        # fan-out form used by gemma3_causallm.h; the others are declared here
        # and defined in the .cpp.
        if hook == "setupParameters":
            c_decls.append(
                "  // Disambiguates the diamond: CausalLM and "
                f"{tcls} both override\n"
                "  // this, so neither dominates and the most-derived class "
                "must provide\n"
                "  // the final overrider. Fans out to both bases, base first.\n"
                "  void setupParameters(json &cfg, json &generation_cfg,\n"
                "                       json &nntr_cfg) override {\n"
                "    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);\n"
                f"    {tcls}::setupParameters(cfg, generation_cfg, nntr_cfg);\n"
                "  }"
            )
        else:
            c_decls.append("  " + _declaration(sig))

    # No explicit setupParameters() call in the constructor: the inline
    # fan-out override above is the mechanism, matching gemma3_causallm.h.
    # Calling it from the ctor as well would run CausalLM::setupParameters
    # twice, since the CausalLM constructor already invokes it.
    setup_call = ""

    banner = _SPDX_H.format(
        filename=f"{stem}_causallm.h", date=date, arch=arch,
        model_id=plan.get("model_id", "<unknown>"),
        override_notes=_override_notes(plan),
    )

    t_body = "\n\n".join(t_decls) if t_decls else \
        "  // No architecture-specific overrides needed."
    c_body = "\n\n".join(c_decls) if c_decls else ""
    c_section = f"\n{c_body}\n" if c_body else ""

    return f"""{banner}
#ifndef {guard}
#define {guard}

#include <causal_lm.h>

namespace causallm {{

/**
 * @brief {arch} transformer body.
 *
 * Inherits Transformer *virtually* so {ccls} can initialise the single shared
 * Transformer base itself -- see the {ccls} constructor below.
 */
class {tcls} : virtual public Transformer {{
public:
  static constexpr const char *architectures = "{tcls}";

  {tcls}(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {{}}

  virtual ~{tcls}() = default;

{t_body}
}};

/**
 * @brief {arch} causal LM.
 *
 * Diamond inheritance: both CausalLM and {tcls} derive virtually from
 * Transformer, so this most-derived class MUST initialise Transformer first,
 * and with ModelType::CAUSALLM. The base initialiser list order below is
 * required, not stylistic.
 */
class {ccls} : public CausalLM, public {tcls} {{
public:
  static constexpr const char *architectures = "{arch}";

  {ccls}(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    {tcls}(cfg, generation_cfg, nntr_cfg) {{{setup_call}}}

  virtual ~{ccls}() = default;
{c_section}}};

}} // namespace causallm

#endif // {guard}
"""


def emit_source(plan: Dict, date: str = "2025") -> Tuple[str, List[Dict]]:
    """
    Emit the .cpp skeleton.

    Returns (source_text, holes) where each hole carries the marker id, hook
    name, and the plan section the body should be transcribed from. Method
    signatures are written here in full; only bodies are left open.
    """
    tcls = plan["transformer_class"]
    ccls = plan["class_name"]
    stem = plan["file_stem"]
    arch = plan.get("architecture", "")

    # Bare includes, matching every shipped model (qwen2/qwen3/gemma3 all use
    # #include <causal_lm.h>, <app_context.h>, etc. with no "nntrainer/"
    # prefix -- that prefix doesn't match any of the project's actual -I
    # search paths and made the generated file fail to compile).
    includes = [
        "#include <app_context.h>",
        "#include <engine.h>",
        "#include <llm_util.hpp>",
        "#include <model.h>",
        f"#include <{stem}_causallm.h>",
    ]
    includes += [f"#include <{h}>" for h in _custom_headers(plan)]

    banner = _SPDX_H.format(
        filename=f"{stem}_causallm.cpp", date=date, arch=arch,
        model_id=plan.get("model_id", "<unknown>"),
        override_notes=_override_notes(plan),
    )

    bodies: List[str] = []
    holes: List[Dict] = []
    hid = 0

    for h in plan.get("holes", []):
        hook = h["hook"]
        sig = _SIGNATURES.get(hook)
        if not sig:
            continue

        # registerCustomLayers is fully determined -- emit it, do not ask.
        if hook == "registerCustomLayers":
            continue

        hid += 1
        decl = _definition(sig, tcls)
        ret_hint = ("  // Must end with: return <final Tensor>;"
                    if sig["ret"] == "Tensor" else
                    "  // Must end with: return {x, h};"
                    if sig["ret"].startswith("std::pair") else
                    "  // Returns void.")
        section = _plan_section_for(hook)
        bodies.append(
            f"{decl}\n"
            f"  // ===FILL:{hid}===\n"
            f"  // hook : {hook}\n"
            f"  // why  : {h['goal']}\n"
            f"  // plan : resolved.{section}\n"
            f"{ret_hint}\n"
            f"  // Replace this whole comment block with the method body.\n"
            f"  // ===END:{hid}===\n"
            f"}}"
        )
        holes.append({
            "id": hid, "hook": hook, "goal": h["goal"],
            "layers": h.get("layers", []), "section": section,
        })

    # XCausalLM::constructModel -- the diamond disambiguator. Deterministic:
    # delegate to the architecture's override, then append the standard LM
    # head. Emitted rather than asked for, because delegating to the wrong
    # base here silently discards every architecture override.
    if any(h["hook"] == "constructModel" for h in plan.get("holes", [])):
        bodies.append(
            f"std::pair<Tensor, Tensor> {ccls}::constructModel() {{\n"
            f"{_LM_HEAD_APPEND.format(tcls=tcls)}}}"
        )

    # registerCustomLayers: deterministic, on both classes.
    if plan.get("needs_registration") or any(
        h["hook"] == "registerCustomLayers" for h in plan.get("holes", [])
    ):
        bodies.append(
            f"void {tcls}::registerCustomLayers() {{\n"
            f"{_registration_body(plan, tcls)}}}"
        )
        bodies.append(
            f"void {ccls}::registerCustomLayers() {{\n"
            f"  // The diamond means neither base is called automatically --\n"
            f"  // both must be invoked explicitly.\n"
            f"  CausalLM::registerCustomLayers();\n"
            f"  {tcls}::registerCustomLayers();\n"
            f"}}"
        )

    body_text = "\n\n".join(bodies) if bodies else \
        "// This architecture matches the base graph exactly; no overrides."

    source = f"""{banner}
{chr(10).join(includes)}

#include <iostream>
#include <stdexcept>

namespace causallm {{

{body_text}

}} // namespace causallm
"""
    return source, holes


def _plan_section_for(hook: str) -> str:
    return {
        "createAttention": "attention",
        "createMlp": "mlp",
        "createTransformerDecoderBlock": "block",
        "constructModel": "construct",
        "setupParameters": "setup_extra",
    }.get(hook, hook)


def emit_skeletons(plan: Dict) -> Dict:
    """
    Emit both files plus the hole list.

    The returned `holes` replaces plan["holes"] for prompting purposes: it is
    filtered to the holes that actually appear as markers (registerCustomLayers
    is emitted deterministically and therefore removed), and renumbered to
    match the markers in the source.
    """
    header = emit_header(plan)
    source, holes = emit_source(plan)
    return {
        "header": header,
        "source": source,
        "holes": holes,
        "header_filename": f"{plan['file_stem']}_causallm.h",
        "source_filename": f"{plan['file_stem']}_causallm.cpp",
    }
