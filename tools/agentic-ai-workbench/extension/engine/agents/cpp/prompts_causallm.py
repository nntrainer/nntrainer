"""
Prompt templates for generating CausalLM model files.

Design brief: this must work on WEAK models
-------------------------------------------
Contributors will not all have a frontier model. So the prompt is built around
one idea: **weak models are good at copying and transforming, bad at recalling
and synthesising.** Every section below exists to convert a synthesis task into
a transformation task.

Five concrete techniques, and what each removes:

1. RESOLVED PLAN (section 1)
   Python has already decided which layers, in which order, with which property
   values -- including all arithmetic (head_dim, gqa, ffn width). The model
   never computes anything. Removes: arithmetic errors, architecture
   misreadings, "does this model tie embeddings?" guesswork.

2. SKELETON WITH NUMBERED HOLES (section 2)
   Include guard, SPDX header, namespace, the virtual-base constructor diamond,
   and the class declarations are emitted by *us*, deterministically. The model
   fills method bodies only. Removes: the single most error-prone part of the
   file, since the diamond ctor is easy to get wrong and impossible to guess.

3. CLOSED MENU (section 3)
   An explicit table of the exact `createLayer()` type strings, narrowed to the
   layers this plan actually needs. The model picks a string to copy; it never
   names a layer from memory. Removes: hallucinated layer types and property
   keys -- the dominant failure mode.

4. INLINE NEGATIVE EXAMPLES (sections 3 and 4)
   Counter-examples sit next to the rule they protect, not in a separate
   "common mistakes" section. Weak models do not cross-reference sections, so a
   warning three screens away from the decision point is a warning that will
   not fire.

5. MECHANICAL SELF-CHECK (section 6)
   A checklist executable by string matching rather than reasoning ("for each
   createLayer in your output, find its type string in the menu above"). Weak
   models can verify far more reliably than they can plan.

The one exception to "override only deltas": nothing here asks for a standalone
program. `Transformer`/`CausalLM` have ZERO pure virtuals and already build the
entire llama-shaped graph, so a correct output is often 60-100 lines. Asking a
small model for less is what makes it succeed.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from knowledge.causallm_kb import (
    render_catalog,
    render_config_members,
    render_forbidden,
    render_hooks,
    render_name_contract,
    unregistered_types,
)

# ---------------------------------------------------------------------------
# The gap marker. Emitted when an op has no implementing layer.
#
# Two parts on purpose: a human block a contributor can read in their editor,
# and a single-line JSON sentinel the validator greps out into state.json so
# the UI (and later, auto-fix) can consume gaps without re-parsing C++.
# ---------------------------------------------------------------------------
GAP_MARKER_SPEC = """\
// ===================== NNTRAINER-WORKBENCH GAP =====================
// op            : <hf op or module name>
// reason        : no layer in Applications/CausalLM/layers/ implements this
// expected io   : <n> input(s) -> <n> output(s)
// to contribute : 1. add layers/<name>_layer.{h,cpp}
//                 2. give it  static constexpr const char *type = "<name>";
//                 3. add it to layers/meson.build
//                 4. registerFactory it in registerCustomLayers()
//                 5. re-run generation -- this marker disappears
// @workbench-gap {"op":"<op>","suggested_type":"<name>","inputs":<n>,"outputs":<n>,"after":"<preceding layer name>"}
// ===================================================================
"""

# ---------------------------------------------------------------------------
# System prompt. Short and absolute -- long system prompts get truncated in
# small models' attention. Everything conditional lives in the user turn.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You generate C++ model files for the nntrainer CausalLM application
(Applications/CausalLM/models/).

You are a TRANSCRIBER, not a designer. The architecture has already been
resolved for you into a PLAN. Your job is to render that plan into C++ by
filling marked holes in a skeleton.

Four absolute rules:

1. NEVER invent a layer type, property key, method name, or header. If you
   need something that is not in the LAYER MENU, emit the GAP MARKER instead.
2. NEVER compute a value. Every number you need is already in the PLAN. Copy
   it. Do not derive head_dim, gqa size, or ffn width yourself.
3. NEVER write code outside the marked holes. The skeleton is fixed.
4. Layer NAMES are a wire format, not labels. Weight loading matches them
   against safetensors keys. A wrong name compiles fine and then silently
   loads zero weights. Copy names from the plan exactly.

Output only code, in the exact fenced format requested. No prose, no
explanation, no markdown headings.
"""


# ---------------------------------------------------------------------------
# The worked example. ONE example, taken verbatim from a shipping model.
#
# Deliberately not three examples: small models blend multiple examples into
# a hybrid that matches none of them. One canonical example, maximally similar
# to the target, outperforms a diverse set.
# ---------------------------------------------------------------------------
WORKED_EXAMPLE = '''\
This is `Qwen3Transformer::createAttention` from
Applications/CausalLM/models/qwen3/qwen3_causallm.cpp, verbatim and shipping.
It is the closest thing to ground truth in the codebase. Match its shape,
spacing, and comment style.

```cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // Q-reshaped-norm layer (q_norm(q_proj.view(hidden_shape)))
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor q_normed = q_norm(q);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // K-reshaped-norm layer (k_norm(k_proj.view(hidden_shape)))
  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor k_normed = k_norm(k);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // External KV cache placeholders (per-layer). Storage is owned by the host
  // (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}
```

Note four things you must reproduce:
  * `LayerHandle h(createLayer(type, {props}));` then `Tensor t = h(input);`
    -- graph edges come from CALLING the handle, never from an `input_layers`
    property.
  * `withKey("unit", head_dim * n_heads)` passes an int directly; `withKey`
    streams any type. You do not need std::to_string everywhere, but it is
    always safe.
  * KV cache comes from `createKVCachePlaceholders(layer_id, n_heads)` and is
    destructured with `auto [cache_k, cache_v]`. Never hand-build those.
  * `mha({q, k, v, cache_k, cache_v})` -- five inputs, that exact order.
'''


# ---------------------------------------------------------------------------
# Hard constraints. Assembled from the KB so prompt and validator cannot drift.
# ---------------------------------------------------------------------------
def _hard_constraints(plan: Dict) -> str:
    needs_registration = [
        t for t in plan.get("layer_types", []) if t in unregistered_types()
    ]
    reg_block = ""
    if needs_registration:
        cls_names = {
            "reshaped_rms_norm": "causallm::ReshapedRMSNormLayer",
            "rms_reverse_norm": "causallm::RMSReverseNormLayer",
            "qkv_layer": "causallm::QKVLayer",
        }
        lines = "\n".join(
            f"      app_context->registerFactory("
            f"nntrainer::createLayer<{cls_names.get(t, t)}>);"
            for t in needs_registration
        )
        reg_block = f"""
C6. THESE LAYERS ARE NOT REGISTERED BY THE BASE CLASS: {", ".join(needs_registration)}
    createLayer() will THROW at runtime unless you register them first. Your
    registerCustomLayers() override MUST contain:

{lines}

    wrapped in try/catch (std::invalid_argument) -- registering an
    already-taken key throws, and that is expected on re-entry.
"""

    return f"""\
C1. LAYER NAMES ARE A WIRE FORMAT.
    Weight loading matches these against safetensors header keys. The Python
    weight converters emit exactly these names -- NOT HuggingFace dotted names.

{render_name_contract()}

    WRONG: withKey("name", "model.layers.0.self_attn.q_proj")
     RIGHT: withKey("name", "layer" + std::to_string(layer_id) + "_wq")

    A wrong name compiles cleanly, loads zero weights, and produces garbage
    output with no error. This is the most damaging mistake you can make.

C2. THESE VALUES ARE ALREADY MEMBERS. Use them; never re-read the json, never
    recompute them.

{render_config_members()}

C3. PROPERTY KEYS ARE CLOSED PER LAYER.
    Use only keys listed in that layer's menu entry. Layers marked
    `[+ weight_initializer, ...]` also accept the LayerImpl set. Layers WITHOUT
    that marker are plain nntrainer::Layer subclasses and will THROW
    std::invalid_argument on those keys.

    WRONG: createLayer("rms_norm", {{withKey("weight_initializer", "ones")}})
              -> throws at runtime; rms_norm is a plain Layer
     RIGHT: createLayer("rms_norm", {{withKey("epsilon", ...),
                                      withKey("packed", "false")}})

C4. THESE SYMBOLS DO NOT EXIST. Every one appeared in earlier broken output.

{render_forbidden()}

C5. SWIGLU INPUT ORDER IS INVERTED ON PURPOSE.
     RIGHT: Tensor act = swiglu({{up, gate}}, {{1, 0}});
    WRONG: swiglu({{gate, up}})      -- compiles, wrong numerics, silent
    WRONG: swiglu({{up, gate}})      -- compiles, wrong numerics, silent
    The layer reads input[0] as gate, but nntrainer binaries store MLP weights
    in up,gate order. The {{1, 0}} remap reconciles the two. Always both.
{reg_block}"""


# ---------------------------------------------------------------------------
# Self-check. Mechanical steps only -- each is a string lookup, not a judgement.
# ---------------------------------------------------------------------------
SELF_CHECK = """\
Before you emit anything, run these checks against your own draft. Each is a
lookup, not a judgement call. Fix what fails.

  1. For every `createLayer("X", ...)` in your draft: find "X" verbatim in the
     LAYER MENU. If it is not there, you invented it -> replace that layer with
     the GAP MARKER.

  2. For every `withKey("k", ...)`: find "k" in that layer's `properties` line,
     or in the node-prop list (name, input_shape, weight_dtype, packed,
     shared_from, activation, trainable). If neither -> delete the withKey.

  3. For every `withKey("name", ...)`: find that name pattern in constraint C1.
     If it is not there, or if it contains a '.' character -> it is wrong.

  4. Search your draft for each symbol in constraint C4. Zero hits expected.

  5. If your draft contains `swiglu`: confirm it is exactly
     `swiglu({up, gate}, {1, 0})`.

  6. If your draft contains `mha_core`: confirm the call is
     `mha({q, k, v, cache_k, cache_v})` -- five inputs, that order -- and that
     cache_k/cache_v came from `createKVCachePlaceholders(...)`.

  7. Count your method definitions. It must equal the number of FILL holes in
     the skeleton. Not more: you do not add methods. Not fewer: every hole
     gets filled.

  8. Confirm you emitted no `main()`, no `#include <model.h>` beyond what the
     skeleton already has, and no class declaration -- those live in the .h,
     which is generated for you.
"""


# ---------------------------------------------------------------------------
# Main generation prompt.
# ---------------------------------------------------------------------------
def generation_prompt(
    plan: Dict,
    skeleton_h: str,
    skeleton_cpp: str,
    example: str = WORKED_EXAMPLE,
) -> str:
    """
    Build the first-pass generation prompt.

    plan
        Resolved ModelPlan from agents/cpp/model_plan.py. Authoritative: every
        layer, property value, and name is already decided. Must carry
        `layer_types` (list[str]) so the menu can be narrowed.
    skeleton_h / skeleton_cpp
        Deterministically generated scaffolds containing `// ===FILL:n===`
        markers. The model only produces the hole bodies.
    """
    holes = plan.get("holes", [])
    hole_block = "\n".join(
        f"  FILL:{h['id']}  {h['hook']}\n"
        f"           goal   : {h['goal']}\n"
        f"           layers : {', '.join(h.get('layers', [])) or '(see plan)'}"
        for h in holes
    ) or "  (none -- see skeleton markers)"

    hook_names = [h["hook"] for h in holes if h.get("hook")]

    return f"""\
Fill in {len(holes)} marked hole(s) in the C++ skeleton below. Nothing else.

Model      : {plan.get('model_id', '<unknown>')}
Architecture: {plan.get('architecture', '<unknown>')}
Class      : {plan.get('class_name', '<unknown>')}
Base       : {plan.get('base_class', 'causallm::CausalLM')}

The base classes have ZERO pure virtual methods and already build the complete
llama-shaped graph (input0 -> embedding0 -> N decoder blocks -> output_norm ->
LM head), with rms_norm / fully_connected / swiglu / mha_core wired. You are
overriding ONLY where this architecture deviates. A correct answer here is
short. If you find yourself writing a whole model, you have misread the task.

================================ 1. THE PLAN ================================
Resolved by static analysis of the model config. AUTHORITATIVE -- every layer,
property value, and name below is already correct. Transcribe; do not adjust,
do not recompute, do not "improve".

{json.dumps(plan.get('resolved', plan), indent=2)}

HOLES TO FILL:
{hole_block}

============================== 2. THE SKELETON ==============================
Already written for you: include guard, SPDX header, namespace, class
declarations, and the virtual-base constructor chain. Do not reproduce or
modify any of it. Emit ONLY the bodies for the `// ===FILL:n===` markers.

--- {plan.get('file_stem', 'model')}_causallm.h ---
```cpp
{skeleton_h}
```

--- {plan.get('file_stem', 'model')}_causallm.cpp ---
```cpp
{skeleton_cpp}
```

============================= 3. THE LAYER MENU =============================
The COMPLETE set of layers available to you. Copy type strings from here
character-for-character. If what you need is not here, it does not exist --
emit the GAP MARKER.

{render_catalog(only=plan.get('layer_types'))}

=========================== 4. HARD CONSTRAINTS ============================
{_hard_constraints(plan)}

======================= 5. OVERRIDABLE HOOKS (exact) =======================
These are the ONLY methods you may define. Signatures are exact -- copy them
character-for-character, including `const int layer_id` and parameter order.

{render_hooks(only=hook_names or None)}

=========================== 6. WORKED EXAMPLE =============================
{example}

============================ 7. GAP MARKER ================================
If the plan calls for an operation with no layer in the menu, emit this in
place of that layer instead of inventing one. A marked gap is a useful
contribution; an invented layer is a silent bug.

{GAP_MARKER_SPEC}

============================= 8. SELF-CHECK ===============================
{SELF_CHECK}

============================ 9. OUTPUT FORMAT =============================
Emit exactly two fenced blocks, in this order, with these exact header
comments. No prose before, between, or after them.

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.h
<the complete header, skeleton with holes filled>
```

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.cpp
<the complete source, skeleton with holes filled>
```
"""


# ---------------------------------------------------------------------------
# Correction prompt. Narrow by construction: a weak model handed a full file
# plus errors tends to rewrite everything and regress working code, so this
# restates only the violated constraints and demands a minimal diff.
# ---------------------------------------------------------------------------
def correction_prompt(
    plan: Dict,
    previous_code: str,
    findings: List[Dict],
    attempt: int,
    max_attempts: int,
) -> str:
    """
    findings
        Structured entries from plan_validator / the compiler, each with
        `kind`, `detail`, and ideally `fix`. Structured beats a raw compiler
        dump: it tells a small model what to change instead of asking it to
        infer that from g++ output.
    """
    finding_block = "\n".join(
        f"  [{i + 1}] {f.get('kind', 'error')}: {f.get('detail', '')}"
        + (f"\n        FIX: {f['fix']}" if f.get("fix") else "")
        for i, f in enumerate(findings)
    )

    violated = sorted({f["constraint"] for f in findings if f.get("constraint")})
    constraint_reminder = ""
    if violated:
        constraint_reminder = f"""
You violated these constraints. Re-read them:

{_hard_constraints(plan)}
"""

    return f"""\
Your previous output was rejected (attempt {attempt}/{max_attempts}).

Fix ONLY the numbered problems below. Change nothing else. Code that was not
flagged is correct -- rewriting it will introduce new failures.

============================== PROBLEMS ==============================
{finding_block}
{constraint_reminder}
========================= YOUR PREVIOUS OUTPUT =======================
```cpp
{previous_code}
```

============================ THE PLAN (unchanged) ====================
{json.dumps(plan.get('resolved', plan), indent=2)}

============================== LAYER MENU ============================
{render_catalog(only=plan.get('layer_types'))}

================================ RULES ===============================
  * Minimal diff. Touch only the lines implicated above.
  * If a problem says a layer type does not exist, do NOT substitute a
    similar-sounding one -- emit the GAP MARKER:

{GAP_MARKER_SPEC}

  * Do not add methods, includes, or main().

Re-emit both files in full, same two-fence format as before:

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.h
...
```

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.cpp
...
```
"""


# ---------------------------------------------------------------------------
# Registration snippet prompt. Deliberately separate from model generation.
#
# Registration is three near-identical edits in three unrelated files -- pure
# boilerplate with zero architectural judgement. Splitting it out keeps the
# main prompt focused and means a weak model never has to hold both tasks at
# once. In practice this is generated in Python and never needs an LLM; the
# prompt exists for the case where call sites have drifted.
# ---------------------------------------------------------------------------
def registration_prompt(plan: Dict) -> str:
    cls = plan.get("class_name", "XCausalLM")
    arch = plan.get("architecture", "XForCausalLM")
    stem = plan.get("file_stem", "x")
    return f"""\
Emit the registration edits for a new CausalLM model. Boilerplate only -- no
judgement required.

Class       : causallm::{cls}
Header      : {stem}_causallm.h
Factory key : "{arch}"   (from config.json architectures[0])

This model must be registered in THREE independent translation units. There is
no self-registration; missing one means the model is invisible to that binary.

  1. Applications/CausalLM/main.cpp              -- top of main(), + the #include
  2. Applications/CausalLM/api/causal_lm_api.cpp -- inside register_models()
  3. Applications/CausalLM/quantize.cpp          -- inside registerAllModels()

The shape, from the shipping Qwen3 registration:

```cpp
  causallm::Factory::Instance().registerModel(
    "{arch}", [](json cfg, json generation_cfg, json nntr_cfg) {{
      return std::make_unique<causallm::{cls}>(cfg, generation_cfg,
                                               nntr_cfg);
    }});
```

Note the lambda takes `json` BY VALUE even though Factory::Creator declares
`json &` -- std::function permits the conversion, and every shipping call site
is written this way. Do not "fix" it to a reference.

Also emit the two meson edits:

  Applications/CausalLM/models/{stem}/meson.build   (new file)
```meson
{stem}_src = [
    meson.current_source_dir() / '{stem}_causallm.cpp',
]

{stem}_inc = include_directories('.')

causallm_src += {stem}_src
causallm_inc += {stem}_inc
```

  Applications/CausalLM/models/meson.build  -- add one line:
```meson
subdir('{stem}')
```

Output each edit as a fenced block preceded by `// FILE: <path>`. For edits to
existing files, show only the added lines plus two lines of surrounding
context. No prose.
"""
