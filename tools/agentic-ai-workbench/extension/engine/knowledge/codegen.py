# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file codegen.py
# @brief Generates nntrainer CausalLM C++ for a Hugging Face model with the Claude CLI.
# @author prachi-t17 <prachi.t@samsung.com>

# This module holds one whole workbench step, kept together on purpose.
# pylint: disable=too-many-lines

"""
Generate nntrainer CausalLM C++ for a Hugging Face model with the Claude CLI,
using the code bank as context.

  review = generate(facts, bank, llm, out_root, ...)

1. Checklist + plan (checklist.py): for each hook the bank decides
     INHERIT  a verified base implementation fits -> do not override
     COPY     a verified (or upstream-approved) model hook fits -> Claude must
              copy it verbatim; the validator enforces that
     WRITE    nothing fits -> Claude writes it, starting from the closest bank
              code and the Hugging Face reference implementation
2. Claude CLI writes the .h/.cpp (and the weight converter when no verified
   one fits). codegen_validator.py checks the reply; errors go back to Claude
   (up to `max_attempts`).
3. Build glue that is not model logic (meson.build, the registration
   function, res/ configs) is written deterministically.
4. Everything lands in tools/agentic-ai-workbench/generated/<slug>/ with a
   review.json. New parts are added to the bank as `generated`; the user's
   approval (Review tab) moves them to `approved`, and Verify's build + HF
   comparison to `verified` -- only then are they reused.

A model that a whole-model box explains (Gemma4, LFM2) is copied from the bank
with renamed classes instead of being retyped by the LLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import causallm_kb as kb
from .checklist import (
    ChecklistReport,
    ModelFacts,
    STRUCTURAL_HOOKS,
    guess_hook,
    match_bank,
    run_checklist,
)
from .code_bank import CodeBank, workbench_dir
from .codegen_validator import ValidationResult, validate
from .cpp_extract import find_classes, find_layer_types
from .checklist import FeatureResult, derived_facts
from . import compile_check
from . import verify  # verify imports this module too; both only use each other at call time

FRAMEWORK_CLASSES = {"Transformer", "CausalLM", "SentenceTransformer"}
LLM = Callable[[str, str], str]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_FILE_RE = re.compile(r"<<<FILE:\s*([^>\n]+?)\s*>>>[ \t]*\n(.*?)\n?<<<END FILE>>>", re.S)

HF_CLASS_FOR_HOOK = {
    "createAttention": r"Attention$",
    "createMlp": r"(MLP|Mlp|MoeBlock|MoE|Experts|FeedForward)$",
    "createTransformerDecoderBlock": r"DecoderLayer$",
    "constructModel": r"(Model|ForCausalLM)$",
}


class CodegenError(RuntimeError):
    """Code generation, review or verification cannot go on."""


def generated_root() -> Path:
    """tools/agentic-ai-workbench/generated (or $WORKBENCH_GENERATED_DIR)."""
    env = os.environ.get("WORKBENCH_GENERATED_DIR")
    if env:
        return Path(env).expanduser()
    wb_dir = workbench_dir()
    if wb_dir is None:
        raise CodegenError(
            "cannot find tools/agentic-ai-workbench: set NNTRAINER_ROOT to the nntrainer checkout"
        )
    return wb_dir / "generated"


# ------------------------------------------------------------------ spec
@dataclass
# pylint: disable-next=too-many-instance-attributes
class HookPlan:
    """How one CausalLM hook is produced (inherit, copy or write) and its plan state."""
    hook: str
    action: str                     # inherit | copy | write
    state: str = ""                 # plan state: reuse | needs_verification | new
    entry_id: str = ""
    entry_status: str = ""
    owner_class: str = ""
    code: str = ""
    features: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """True unless the hook's code is reused as it is."""
        return self.state != "reuse"


@dataclass
# pylint: disable-next=too-many-instance-attributes
class GenSpec:
    """Everything the generator needs for one model: names, hooks and classes."""
    model_id: str
    hf_architecture: str
    prefix: str
    slug: str
    transformer_class: str
    causallm_class: str
    header_name: str
    source_name: str
    hooks: Dict[str, HookPlan]
    converter_action: str = "write"          # reuse | write
    converter_entry: str = ""
    converter_status: str = ""
    whole_model: Optional[dict] = None
    tensors: Dict[str, Any] = field(default_factory=dict)
    tensor_counts: Dict[str, int] = field(default_factory=dict)
    tensors_whole_model: bool = False
    known_layer_types: List[str] = field(default_factory=list)
    core_layer_types: List[str] = field(
        default_factory=list
    )  # nntrainer's own (registered by nntrainer)
    layer_classes: Dict[str, str] = field(default_factory=dict)
    learned: Dict[str, dict] = field(default_factory=dict)
    other_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The spec as JSON-ready data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GenSpec":
        """A spec from to_dict() data."""
        data = dict(data)
        data["hooks"] = {k: HookPlan(**v) for k, v in data["hooks"].items()}
        return cls(**data)


def class_prefix(arch: str) -> str:
    """A C++ class name prefix from a Hugging Face architecture name."""
    base = re.sub(r"(ForCausalLM|ForConditionalGeneration|LMHeadModel|Model)$", "", arch or "") or (
        arch or "Model"
    )
    base = re.sub(r"[^A-Za-z0-9]", "", base)
    return (base[:1].upper() + base[1:]) or "Model"


def slug_of(prefix: str) -> str:
    """A snake_case file name stem for a class prefix."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", prefix).lower()


def _layer_info(bank: CodeBank) -> Tuple[List[str], Dict[str, str]]:
    types, classes = set(), {}
    for entry in bank.list(kind="layer", status=("verified", "approved")):
        for name, text in entry["files"].items():
            layer_types = find_layer_types(text)
            if layer_types and name.endswith((".h", ".hpp")):
                class_names = find_classes(text)
                for layer_type in layer_types:
                    types.add(layer_type)
                    if class_names:
                        classes[layer_type] = class_names[0].name
    return sorted(types), classes


# pylint: disable-next=too-many-locals
def make_spec(
    model_id: str, facts: ModelFacts, report: ChecklistReport, plan: dict, bank: CodeBank
) -> GenSpec:
    """The generation spec of a model, from its facts, checklist report and plan."""
    arch = (report.architectures or [report.model_type or "Model"])[0]
    prefix = class_prefix(arch)
    slug = slug_of(prefix)
    transformer_cls, causallm_cls = f"{prefix}WorkbenchTransformer", f"{prefix}WorkbenchCausalLM"
    hooks: Dict[str, HookPlan] = {}
    for hook in STRUCTURAL_HOOKS:
        hdr = plan["hooks"][hook]
        eid = (
            hdr["reuse"]
            if hdr["state"] == "reuse"
            else (
                hdr["unverified_matches"][0]
                if hdr["state"] == "needs_verification" and hdr["unverified_matches"]
                else ""
            )
        )
        hook_plan = HookPlan(
            hook=hook,
            action="write",
            state=hdr["state"],
            features=list(hdr["features"]),
            new=list(hdr.get("new", [])),
            examples=list(hdr.get("examples", [])),
        )
        if eid and hdr["state"] != "new":
            entry = bank.get(eid) or {}
            owner = (entry.get("evidence") or {}).get("class") or entry.get("name", "").split("::")[
                0
            ]
            hook_plan.entry_id, hook_plan.entry_status, hook_plan.owner_class, hook_plan.code = (
                eid,
                entry.get("status", ""),
                owner,
                entry.get("code", ""),
            )
            hook_plan.action = "inherit" if owner in FRAMEWORK_CLASSES else "copy"
        hooks[hook] = hook_plan
    conv = plan["converter"]
    if conv["state"] == "reuse":
        c_action, c_entry = "reuse", conv["reuse"]
    elif conv["state"] == "needs_verification" and conv["unverified_matches"]:
        c_action, c_entry = "reuse", conv["unverified_matches"][0]
    else:
        c_action, c_entry = "write", ""
    c_status = (bank.get(c_entry) or {}).get("status", "") if c_entry else ""
    types, classes = _layer_info(bank)
    learned = {}
    derived = derived_facts(facts)
    for hook in STRUCTURAL_HOOKS:
        spec_ = _learned_feature(report, hook, slug, model_id, derived)
        if spec_:
            learned[hook] = spec_
    other = {
        k: v for k, v in report.unclaimed_config.items() if guess_hook(k) not in STRUCTURAL_HOOKS
    }
    whole = (
        plan.get("whole_model") if (plan.get("whole_model") or {}).get("state") == "reuse" else None
    )
    return GenSpec(
        model_id=model_id,
        hf_architecture=arch,
        prefix=prefix,
        slug=slug,
        transformer_class=transformer_cls,
        causallm_class=causallm_cls,
        header_name=f"{slug}_workbench_causallm.h",
        source_name=f"{slug}_workbench_causallm.cpp",
        hooks=hooks,
        converter_action=c_action,
        converter_entry=c_entry,
        converter_status=c_status,
        whole_model=whole,
        tensors=dict(facts.tensors),
        tensor_counts=dict(facts.tensor_counts),
        tensors_whole_model=bool(whole),
        known_layer_types=types,
        core_layer_types=list(bank.get_meta("nntrainer_core_layer_types", []) or []),
        layer_classes=classes,
        learned=learned,
        other_config=other,
    )


def _symbolic_dim(pattern: str, last: int, derived: Dict[str, Any]) -> Optional[str]:
    """Express a tensor's last dimension relative to the model (so the box fits
    every size of the family): @head_dim, @q_width, @kv_width or @hidden_size."""
    leaf = pattern.split(".")[-2] if pattern.count(".") else pattern
    order = (
        ["head_dim"]
        + (["q_width", "kv_width"] if leaf.startswith("q") else ["kv_width", "q_width"])
        + ["hidden_size"]
    )
    for name in order:
        if derived.get(name) and derived[name] == last:
            return "@" + name
    return None


def _learned_feature(report: ChecklistReport, hook: str, slug: str, model_id: str,
                     derived: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """A checklist box built from the evidence that made `hook` new, saved to
    the bank when the user approves the generated code."""
    unclaimed = [t for t in report.unclaimed_tensors if t["hook"] == hook]
    tensors = [t["pattern"] for t in unclaimed]
    config = {k: v for k, v in report.unclaimed_config.items() if guess_hook(k) == hook}
    if not tensors and not config:
        return None
    when: List[dict] = [{"tensor": t} for t in tensors]
    for tensor in unclaimed:  # shape, relative to the model, when known
        shape = tensor.get("shape")
        dim = _symbolic_dim(tensor["pattern"], shape[-1], derived or {}) if shape else None
        if dim:
            when.append({"shape": tensor["pattern"], "dims": [dim]})
    for k, val in config.items():
        when.append(
            {"config": k, "op": "eq", "value": val}
            if isinstance(val, (int, float, str, bool))
            else {"config": k, "op": "exists"}
        )
    return {
        "id": f"learned_{slug}_{hook}",
        "label": f"{hook} part first generated for {model_id}",
        "hooks": [hook],
        "from": [slug],
        "when": when,
        "claims": {"tensors": tensors, "config": sorted(config)},
        "note": (
            "Learned from an approved generated model; ticks for checkpoints with the same"
            " evidence."
        ),
    }


# ---------------------------------------------------------------- prompt
SYSTEM_RULES = """You write nntrainer CausalLM model code (C++17) for nntrainer's \
Applications/CausalLM. The code is compiled
together with Applications/CausalLM/models/transformer.h and causal_lm.h (shown in the message) \
and must follow
these rules exactly.

CLASSES AND FILES
- Write exactly two classes in namespace causallm, declared in {header} and defined in {source}
  ({source} starts with #include "{header}" and, like every upstream model source, includes \
<llm_util.hpp>
  (withKey, createLayer), <model.h>, <app_context.h> and <engine.h>; {header} includes \
<causal_lm.h>):
    class {T} : virtual public Transformer        // overrides the model hooks
    class {C} : public CausalLM, public {T}       // the class that gets registered
- {T}(json &cfg, json &generation_cfg, json &nntr_cfg) : Transformer(cfg, generation_cfg, \
nntr_cfg) {{ ... }}
- {C}(json &cfg, json &generation_cfg, json &nntr_cfg) :
      Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),   // the virtual base, first
      CausalLM(cfg, generation_cfg, nntr_cfg),
      {T}(cfg, generation_cfg, nntr_cfg) {{ ... }}
- Use the exact hook signatures from transformer.h. Use the base members (DIM, NUM_HEADS, \
HEAD_DIM, GQA_SIZE,
  NUM_LAYERS, ...) instead of literals so one class serves every size of the model.

CONFIG
- Do NOT override setupParameters. An override is never called while the object is being \
constructed (C++
  dispatches to the base class there), so anything read in it is silently lost. Read extra \
config.json keys in
  {T}'s constructor body into members, the way Gemma3Transformer's constructor sets EMBEDDING_SCALE.

HOOKS (the plan in the message says which applies)
- INHERIT: do not override the hook.
- COPY: override the hook with the given body character for character; only the class qualifier \
changes. It is
  verified against Hugging Face; do not change it. A checker compares it. Declare and initialise \
any member it uses.
- WRITE: implement the hook from the Hugging Face reference code, starting from the closest \
existing code shown.
- If {T} overrides registerCustomLayers, {C} must override it too and call \
CausalLM::registerCustomLayers() and
  {T}::registerCustomLayers(). Likewise constructModel: if {T} overrides it, {C} must too (see \
Gemma4CausalLM).
- In registerCustomLayers, register every custom layer type you use that \
Transformer::registerCustomLayers does
  not already register (rms_norm, swiglu, mha_core, tie_word_embeddings and embedding_layer are \
registered there).

LAYERS
- Build graphs only with createLayer("<type>", {{withKey(...), ...}}) and LayerHandle calls, only \
with the layer
  types in the catalogue below (with their listed properties) or nntrainer's own layer types:
  {core}
  (nntrainer registers its own types; for example LayerNorm with bias is "layer_normalization", \
not rms_norm).
- Every createLayer call sets withKey("name", ...) following the naming contract below. \
Weight-bearing layers
  must be created in the order the weight converter writes their weights.
- Prefer existing layers. Only if a part truly cannot be built from them, add a new layer as \
<name>_layer.h and
  <name>_layer.cpp (namespace causallm, `static constexpr const char *type = "<name>";`), modelled \
on an
  existing custom layer, and register it.

WEIGHT CONVERTER (only when the plan asks for one)
- weight_converter.py reads the Hugging Face safetensors and writes nntrainer's .bin: one tensor \
after another in
  exactly the order the layers are created (embedding; per decoder layer in creation order; final \
norm; lm_head
  unless tied), fully connected weights transposed. Follow the example converter's structure and \
CLI, and keep
  its entry point save_<model>_for_nntrainer(params, n_layers, dtype, file) (or (params, config, \
dtype, file))
  at module level: the verifier calls it on a tiny random-weight copy of the model.

NEVER USE
{forbidden}

OUTPUT FORMAT
Reply with the files only, each wrapped exactly like this, and nothing else before, between or \
after them:
<<<FILE: file_name>>>
...complete file content...
<<<END FILE>>>

NAMING CONTRACT
{names}

BASE MEMBERS
{members}

HOOKS
{hooks}

LAYER CATALOGUE
{catalog}
"""


def build_system_prompt(spec: GenSpec) -> str:
    """The system prompt for Claude for this spec."""
    core = ", ".join(spec.core_layer_types) or "(list not available)"
    return SYSTEM_RULES.format(
        header=spec.header_name,
        source=spec.source_name,
        T=spec.transformer_class,
        C=spec.causallm_class,
        core=core,
        forbidden=kb.render_forbidden(),
        names=kb.render_name_contract(),
        members=kb.render_config_members(),
        hooks=kb.render_hooks(),
        catalog=kb.render_catalog(),
    )


def _fence(lang: str, text: str) -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _model_files_for(bank: CodeBank, hook_entry_id: str) -> Dict[str, str]:
    entry = bank.get(hook_entry_id) or {}
    model_entry = bank.get(f"model:{entry.get('architecture', '')}")
    if model_entry:
        return {k: v for k, v in model_entry["files"].items() if k.endswith((".h", ".cpp"))}
    hdr = (entry.get("evidence") or {}).get("header_text")
    return {entry.get("evidence", {}).get("header", "header.h"): hdr} if hdr else {}


def _closest_model(bank: CodeBank, report: ChecklistReport) -> Optional[dict]:
    ticked = {f.id for f in report.ticked}
    best, score = None, -1.0
    for model_entry in bank.list(kind="model", status=("verified",)):
        feats = set(model_entry["features"]) | set(model_entry["supports"])
        if not model_entry["features"]:
            continue
        similarity = len(ticked & feats) / max(len(ticked | feats), 1)
        if similarity > score:
            best, score = model_entry, similarity
    return best


# pylint: disable-next=too-many-locals,too-many-branches
def build_user_prompt(spec: GenSpec, report: ChecklistReport, bank: CodeBank, raw_config: dict,
                      hf_sources: Dict[str, str]) -> str:
    """The task prompt for Claude: the model, its features, config, examples and Hugging Face
    sources."""
    labels = {f.id: f.label for f in report.features}
    out = [
        f"# Task\nGenerate nntrainer CausalLM code for `{spec.model_id}` ({spec.hf_architecture}).",
        f"Return: {spec.header_name}, {spec.source_name}"
        + (", weight_converter.py" if spec.converter_action == "write" else "")
        + " (plus new layer files only if a new layer is truly needed).",
        f"Classes: {spec.transformer_class} and {spec.causallm_class} "
        f"(registered as \"{spec.hf_architecture}\" by separate glue code you do not write).",
        "",
        "# Plan -- follow it exactly",
    ]
    deps_shown = set()
    for hook, hook_plan in spec.hooks.items():
        feats = ", ".join(labels.get(f, f) for f in hook_plan.features) or "(none detected)"
        if hook_plan.action == "inherit":
            out.append(
                f"\n## {hook}: INHERIT\nDo not override {hook}. The base implementation"
                f" ({hook_plan.entry_id}, {hook_plan.entry_status}) already implements: {feats}."
            )
        elif hook_plan.action == "copy":
            out.append(
                f"\n## {hook}: COPY\nOverride {hook} with exactly this"
                f" {hook_plan.entry_status} code from {hook_plan.owner_class} (change only the"
                f" class qualifier to the class you put it in). It implements: {feats}.\n"
                + _fence("cpp", hook_plan.code)
            )
            files = _model_files_for(bank, hook_plan.entry_id)
            if files and hook_plan.owner_class not in deps_shown:
                deps_shown.add(hook_plan.owner_class)
                out.append(
                    "Members and layer registrations this code relies on, from"
                    f" {hook_plan.owner_class}'s model (declare/initialise/register what it needs):"
                )
                for name, text in files.items():
                    out.append(f"### {name}\n" + _fence("cpp", text))
        else:
            out.append(f"\n## {hook}: WRITE (new code)\nIt must implement: {feats}.")
            if hook_plan.new:
                out.append(
                    "Not covered by any existing code:\n"
                    + "\n".join(f"- {n}" for n in hook_plan.new)
                )
            for ex in hook_plan.examples[:3]:
                entry = bank.get(ex)
                if entry:
                    out.append(
                        f"Closest existing code -- {ex} ({entry['status']}):\n"
                        + _fence("cpp", entry["code"])
                    )
    if spec.other_config:
        out.append(
            "\n# Other config keys no existing code explains\nDecide whether each changes the"
            " computation; if so read it in the constructor and use it in the right hook:\n"
            + "\n".join(f"- {k} = {json.dumps(v)}" for k, v in spec.other_config.items())
        )
    out.append("\n# Weight converter")
    if spec.converter_action == "reuse":
        out.append(
            f"Do not write one: {spec.converter_entry} ({spec.converter_status}) is used. Create"
            " layers in the order it writes tensors:\n"
            + _fence("python", (bank.get(spec.converter_entry) or {}).get("code", ""))
        )
    else:
        out.append("Write weight_converter.py. Closest existing converter:")
        conv = bank.search(
            "weight converter " + " ".join(f.id for f in report.ticked),
            kind="converter",
            statuses=("verified",),
            limit=1,
            embeddings="off",
        )
        if conv:
            out.append(f"### {conv[0]['id']}\n" + _fence("python", conv[0]["code"]))
    if hf_sources:
        out.append("\n# Hugging Face reference implementation")
        for name, src in hf_sources.items():
            out.append(f"### {name}\n" + _fence("python", src))
    else:
        out.append(
            "\n# Hugging Face reference implementation\n(not available in this environment -- rely"
            " on the config, the tensor list and the closest existing code)"
        )
    out.append("\n# Checkpoint tensors (layer / expert index shown as *)\n" + "\n".join(
        f"{t}  {spec.tensors[t]}  x{spec.tensor_counts.get(t, 1)}" for t in sorted(spec.tensors)))
    out.append("\n# config.json\n" + _fence("json", json.dumps(raw_config, indent=1, default=str)))
    framework_files = (bank.get("framework:causallm-base") or {}).get("files", {})
    out.append("\n# Base classes (members and hook signatures)")
    for name in ("models/transformer.h", "models/causal_lm.h"):
        if name in framework_files:
            out.append(f"### {name}\n" + _fence("cpp", framework_files[name]))
    style = _closest_model(bank, report)
    if style and not any(o in deps_shown for o in style["evidence"].get("classes", [])):
        out.append(f"\n# Style reference: {style['id']} (hand-written, verified)")
        for name, text in style["files"].items():
            if name.endswith((".h", ".cpp")) and "embedding" not in name:
                out.append(f"### {name}\n" + _fence("cpp", text))
    return "\n".join(out)


def retry_prompt(base: str, errors: List[str], files: Dict[str, str]) -> str:
    """The prompt again, with the failed checks and the previous files."""
    prev = "\n".join(f"<<<FILE: {n}>>>\n{t}\n<<<END FILE>>>" for n, t in files.items())
    return (
        base
        + "\n\n# Your previous answer failed these checks\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\nFix every one and return all files again in the same format.\n\n# Your previous"
        " files\n"
        + prev
    )


def parse_files(reply: str) -> Dict[str, str]:
    """The files in a Claude reply (its <<<FILE: name>>> blocks)."""
    files: Dict[str, str] = {}
    for match in _FILE_RE.finditer(reply or ""):
        name = os.path.basename(match.group(1).strip().replace("\\", "/"))
        if not _SAFE_NAME.match(name):
            continue
        body = match.group(2)
        fence = re.match(r"^\s*```[\w+-]*\n(.*?)\n```\s*$", body, re.S)
        if fence:
            body = fence.group(1)
        files[name] = body.rstrip() + "\n"
    return files


# ------------------------------------------------------------------ glue
def glue_files(
    spec: GenSpec,
    files: Dict[str, str],
    raw_config: dict,
    generation_config: Optional[dict],
    register: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """The registration and build files that go with the generated model files."""
    cpps = [n for n in files if n.endswith(".cpp")]
    reg_name = f"register_{spec.slug}_workbench.cpp"
    register = register or {spec.hf_architecture: spec.causallm_class}
    include = (
        spec.header_name
        if spec.header_name in files
        else next((n for n in files if n.endswith(".h")), "")
    )
    lines = [f"    meson.current_source_dir() / '{n}'," for n in sorted(cpps) + [reg_name]]
    meson = (
        f"# Generated by the Agentic AI Workbench for {spec.model_id}.\n"
        f"{spec.slug}_workbench_src = [\n" + "\n".join(lines) + "\n]\n\n"
        f"causallm_src += {spec.slug}_workbench_src\ncausallm_inc += include_directories('.')\n"
    )
    regs = "\n".join(
        f'  Factory::Instance().registerModel(\n    "{arch}", [](json cfg, json generation_cfg,'
        f' json nntr_cfg) {{\n      return std::make_unique<{cls}>(cfg, generation_cfg,'
        ' nntr_cfg);\n    });'
        for arch, cls in register.items()
    )
    reg = (
        f"// SPDX-License-Identifier: Apache-2.0\n/**\n * @file   {reg_name}\n * @brief  Registers"
        f" the generated {spec.prefix} model (Agentic AI Workbench).\n * @note   Call"
        f" causallm::register{spec.prefix}Workbench() after the built-in registrations;\n *        "
        " it replaces any earlier registration under the same architecture name.\n */\n#include"
        f" <factory.h>\n\n#include \"{include}\"\n\nnamespace causallm {{\n\nvoid"
        f" register{spec.prefix}Workbench() {{\n{regs}\n}}\n\n}} // namespace causallm\n"
    )
    nntr = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": f"nntr_{spec.slug}_fp32.bin",
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "lmhead_dtype": "FP32",
        "bad_word_ids": [],
        "fsu": False,
        "num_to_generate": 64,
        "init_seq_len": 64,
        "max_seq_len": 512,
        "batch_size": 1,
        "tokenizer_file": "tokenizer.json",
    }
    gen = generation_config or {
        k: raw_config[k]
        for k in ("bos_token_id", "eos_token_id", "pad_token_id")
        if k in raw_config
    }
    return {"meson.build": meson, reg_name: reg,
            "res/config.json": json.dumps(raw_config, indent=2, default=str) + "\n",
            "res/generation_config.json": json.dumps(gen, indent=2, default=str) + "\n",
            "res/nntr_config.json": json.dumps(nntr, indent=2) + "\n"}


# pylint: disable-next=too-many-locals
def copy_whole_model(spec: GenSpec, bank: CodeBank) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Files of a whole-model match, classes renamed to <Name>Workbench...
    Returns (files, {architecture: renamed class})."""
    model_entry = bank.get(spec.whole_model["entry"])
    if not model_entry:
        raise CodegenError(f"{spec.whole_model['entry']} is not in the code bank")
    classes = model_entry["evidence"].get("classes", [])

    def rename(code: str) -> str:
        for suffix in ("Transformer", "CausalLM"):
            if code.endswith(suffix) and code != suffix:
                return code[:-len(suffix)] + "Workbench" + suffix
        return code + "Workbench"

    cmap = {c: rename(c) for c in classes}
    fmap = {}
    for path in model_entry["files"]:
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext in (".h", ".cpp"):
            fmap[os.path.basename(path)] = f"{stem}_workbench{ext}"
    out = {}
    for path, text in model_entry["files"].items():
        base = os.path.basename(path)
        if base not in fmap:
            continue
        for old, new in sorted(fmap.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(
                rf'#include\s*([<"]){re.escape(old)}([>"])', rf"#include \g<1>{new}\g<2>", text
            )
            stem_old, stem_new = os.path.splitext(old)[0].upper(), os.path.splitext(new)[0].upper()
            text = re.sub(
                rf"\b(__)?{re.escape(stem_old)}(_H__|_H_|_H)\b", rf"\g<1>{stem_new}\g<2>", text
            )
        for old, new in sorted(cmap.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        out[fmap[base]] = text
    regs = {
        a: cmap.get(c, c)
        for a, c in model_entry["evidence"].get("registered_architectures", {}).items()
    }
    return out, regs


# -------------------------------------------------------------- generate
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def hf_snippet(hook: str, hf_sources: Dict[str, str]) -> str:
    """The Hugging Face source of the class that matches a hook."""
    regex = HF_CLASS_FOR_HOOK.get(hook)
    if not regex:
        return ""
    return "\n\n".join(src for name, src in hf_sources.items() if re.search(regex, name))


# pylint: disable-next=too-many-locals,too-many-arguments
def generate(facts: ModelFacts, bank: CodeBank, llm: Optional[LLM], out_root: Optional[Path] = None,
             raw_config: Optional[dict] = None, generation_config: Optional[dict] = None,
             hf_sources: Optional[Dict[str, str]] = None, model_id: Optional[str] = None,
             max_attempts: int = 3, log: Callable[[str], None] = print) -> dict:
    """Generate, check and save CausalLM code for a model; returns the review."""
    hf_sources = hf_sources or {}
    raw_config = raw_config if raw_config is not None else dict(facts.config)
    model_id = model_id or facts.name
    report = run_checklist(facts, bank.learned_features())
    plan = match_bank(report, bank)
    spec = make_spec(model_id, facts, report, plan, bank)
    out_dir = Path(out_root or generated_root()) / spec.slug
    work = out_dir / ".codegen"
    work.mkdir(parents=True, exist_ok=True)
    (work / "spec.json").write_text(
        json.dumps(spec.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (work / "checklist.json").write_text(
        json.dumps({"report": report.to_dict(), "plan": plan}, indent=2, default=str),
        encoding="utf-8",
    )
    (work / "hf_sources.json").write_text(json.dumps(hf_sources, indent=1), encoding="utf-8")

    register = None
    attempts = 0
    if spec.whole_model:
        files, register = copy_whole_model(spec, bank)
        validation = ValidationResult()
        generator = "copied from the code bank (whole-model match)"
        log(
            f"Whole-model match {spec.whole_model['entry']}: copying the verified files with"
            " renamed classes"
        )
    else:
        if llm is None:
            raise CodegenError("the Claude CLI is not available")
        system = build_system_prompt(spec)
        user = build_user_prompt(spec, report, bank, raw_config, hf_sources)
        (work / "system_prompt.txt").write_text(system, encoding="utf-8")
        (work / "user_prompt.txt").write_text(user, encoding="utf-8")
        files, validation, attempts = _llm_loop(
            spec, bank, llm, system, user, work, max_attempts, log
        )
        generator = "claude-cli"
    review = _finalise(
        out_dir,
        spec,
        report,
        plan,
        files,
        validation,
        bank,
        hf_sources,
        raw_config,
        generation_config,
        register,
        source="generated",
        attempts=attempts,
        generator=generator,
        previous=None,
    )
    return review


# pylint: disable-next=too-many-arguments
def _llm_loop(
    spec: GenSpec,
    bank: CodeBank,
    llm: LLM,
    system: str,
    user: str,
    work: Path,
    max_attempts: int,
    log: Callable[[str], None],
) -> Tuple[Dict[str, str], ValidationResult, int]:
    prompt, files, vres = user, {}, ValidationResult(errors=["no reply"])
    for attempt in range(1, max_attempts + 1):
        log(
            f"Claude CLI: generating {spec.header_name} / {spec.source_name} (attempt"
            f" {attempt}/{max_attempts})"
        )
        reply = llm(system, prompt)
        (work / f"reply_{attempt}.txt").write_text(reply or "", encoding="utf-8")
        got = parse_files(reply)
        if not got:
            vres = ValidationResult(errors=["the reply contained no <<<FILE: ...>>> blocks"])
        else:
            files = got
            vres = validate(files, spec, bank)
        (work / f"validation_{attempt}.json").write_text(
            json.dumps(vres.to_dict(), indent=2), encoding="utf-8"
        )
        if vres.passed:
            log(f"Generated code passed every check on attempt {attempt}")
            return files, vres, attempt
        log(f"Attempt {attempt}: {len(vres.errors)} problem(s): " + "; ".join(vres.errors[:4]))
        prompt = retry_prompt(user, vres.errors, files) if files else user
    return files, vres, max_attempts


def _definition_text(vres: ValidationResult, spec: GenSpec, hook: str) -> Tuple[str, str]:
    for cls in (spec.transformer_class, spec.causallm_class):
        data = vres.definitions.get(cls, {}).get(hook)
        if data:
            return cls, data.text
    return "", ""


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
def _finalise(
    out_dir: Path,
    spec: GenSpec,
    report: ChecklistReport,
    plan: dict,
    files: Dict[str, str],
    vres: ValidationResult,
    bank: CodeBank,
    hf_sources: Dict[str, str],
    raw_config: dict,
    generation_config: Optional[dict],
    register: Optional[Dict[str, str]],
    source: str,
    attempts: int,
    generator: str,
    previous: Optional[dict],
) -> dict:
    files = dict(files)
    if spec.converter_action == "reuse" and spec.converter_entry:
        files["weight_converter.py"] = (bank.get(spec.converter_entry) or {}).get("code", "")
    if not spec.whole_model and files.get(spec.source_name):
        files[spec.source_name] = ensure_common_includes(files[spec.source_name], spec.header_name)
    all_files = dict(files)
    all_files.update(glue_files(spec, files, raw_config, generation_config, register))
    for name, text in all_files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    prev_parts = {p["id"]: p for p in (previous or {}).get("parts", [])}
    parts: List[dict] = []
    header_text = files.get(spec.header_name, "")
    for hook, hook_plan in spec.hooks.items():
        cls, text = _definition_text(vres, spec, hook)
        part = {
            "id": hook,
            "kind": "hook",
            "hook": hook,
            "action": hook_plan.action,
            "plan_state": hook_plan.state,
            "needs_review": hook_plan.needs_review,
            "source_entry": hook_plan.entry_id,
            "source_status": hook_plan.entry_status,
            "class": cls or (hook_plan.owner_class if hook_plan.action == "inherit" else ""),
            "file": "" if hook_plan.action == "inherit" else spec.source_name,
            "features": hook_plan.features,
            "evidence": hook_plan.new,
            "start_from": hook_plan.examples,
            "learned_feature": spec.learned.get(hook),
            "hf_reference": hf_snippet(hook, hf_sources),
            "code": text or (hook_plan.code if hook_plan.action == "inherit" else ""),
            "base_code": (
                hook_plan.code
                if hook_plan.action in ("copy", "inherit")
                else (
                    (bank.get(hook_plan.examples[0]) or {}).get("code", "")
                    if hook_plan.examples
                    else ""
                )
            ),
        }
        if hook_plan.action == "write":
            part["status"] = "generated" if vres.passed else "failed_checks"
            # Keep pointing at the last version that passed, so a later good
            # version can retire it.
            part["entry_id"] = prev_parts.get(hook, {}).get("entry_id", "")
            if vres.passed and text:
                eid = f"hook:{cls}::{hook}@{_sha(text)}"
                if not bank.get(eid):
                    feats = sorted(set(report.ticked_for(hook)) |
                                   ({spec.learned[hook]["id"]} if hook in spec.learned else set()))
                    bank.add_generated(
                        {
                            "id": eid,
                            "kind": "hook",
                            "name": f"{cls}::{hook}",
                            "architecture": spec.slug,
                            "hook": hook,
                            "role": "structural",
                            "features": feats,
                            "code": text,
                            "origin": "workbench",
                            "source": source,
                            "source_ref": f"{spec.model_id} ({generator})",
                            "evidence": {
                                "model": spec.model_id,
                                "class": cls,
                                "file": spec.source_name,
                                "header": spec.header_name,
                                "header_text": header_text,
                                "new": hook_plan.new,
                                "examples": hook_plan.examples,
                            },
                        }
                    )
                old = prev_parts.get(hook, {}).get("entry_id")
                if old and old != eid and (bank.get(old) or {}).get("status") == "generated":
                    bank.reject(old, actor="workbench", reason=f"superseded by {eid}")
                part["entry_id"] = eid
                if prev_parts.get(hook, {}).get("entry_id") == eid:
                    part["status"] = prev_parts[hook].get("status", "generated")
        else:
            part["entry_id"] = hook_plan.entry_id
            part["status"] = (
                "verified" if hook_plan.entry_status == "verified" else "needs_verification"
            )
            if prev_parts.get(hook, {}).get("status") == "accepted":
                part["status"] = "accepted"
        parts.append(part)

    conv_text = files.get("weight_converter.py", "")
    conv = {
        "id": "weight_converter",
        "kind": "converter",
        "hook": "converter",
        "file": "weight_converter.py",
        "action": "copy" if spec.converter_action == "reuse" else "write",
        "code": conv_text,
        "source_entry": spec.converter_entry,
        "source_status": spec.converter_status,
        "needs_review": spec.converter_action != "reuse" or spec.converter_status != "verified",
    }
    if spec.converter_action == "reuse":
        conv["status"] = "verified" if spec.converter_status == "verified" else "needs_verification"
        conv["entry_id"] = spec.converter_entry
    else:
        conv["status"] = "generated" if vres.passed else "failed_checks"
        conv["entry_id"] = prev_parts.get("weight_converter", {}).get("entry_id", "")
        if vres.passed and conv_text:
            eid = f"converter:workbench/{spec.slug}@{_sha(conv_text)}"
            if not bank.get(eid):
                layout = [
                    f.id for f in report.ticked if f.claimed_tensors or "converter" in f.hooks
                ]
                layout += [s["id"] for s in spec.learned.values()]
                bank.add_generated(
                    {
                        "id": eid,
                        "kind": "converter",
                        "name": f"{spec.slug} weight converter",
                        "architecture": spec.slug,
                        "features": sorted(set(layout)),
                        "code": conv_text,
                        "origin": "workbench",
                        "source": source,
                        "source_ref": f"{spec.model_id} ({generator})",
                        "evidence": {"model": spec.model_id},
                    }
                )
            old = conv["entry_id"]
            if old and old != eid and (bank.get(old) or {}).get("status") == "generated":
                bank.reject(old, actor="workbench", reason=f"superseded by {eid}")
            conv["entry_id"] = eid
            if prev_parts.get("weight_converter", {}).get("entry_id") == eid:
                conv["status"] = prev_parts["weight_converter"].get("status", "generated")
    parts.append(conv)

    for name in files:
        if (
            name.endswith((".h", ".cpp"))
            and name not in (spec.header_name, spec.source_name)
            and not spec.whole_model
        ):
            text = files[name]
            types = find_layer_types(text)
            parts.append(
                {
                    "id": f"layer:{name}",
                    "kind": "layer",
                    "hook": "new layer",
                    "file": name,
                    "action": "write",
                    "needs_review": True,
                    "status": "generated" if vres.passed else "failed_checks",
                    "code": text,
                    "layer_types": types,
                    "entry_id": "",
                }
            )
    if vres.passed and not spec.whole_model:
        for stem in sorted({os.path.splitext(n)[0] for n in files if n.endswith(".h")
                            and n not in (spec.header_name,)}):
            layer_files = {n: files[n] for n in files if os.path.splitext(n)[0] == stem}
            types = sorted({t for text in layer_files.values() for t in find_layer_types(text)})
            if not types:
                continue
            eid = f"layer:{types[0]}@{_sha(json.dumps(layer_files, sort_keys=True))}"
            if not bank.get(eid):
                bank.add_generated(
                    {
                        "id": eid,
                        "kind": "layer",
                        "name": stem,
                        "architecture": spec.slug,
                        "files": layer_files,
                        "origin": "workbench",
                        "source": source,
                        "source_ref": f"{spec.model_id} ({generator})",
                        "evidence": {"layer_types": types, "model": spec.model_id},
                    }
                )
            for path in parts:
                if path["kind"] == "layer" and os.path.splitext(path["file"])[0] == stem:
                    path["entry_id"] = eid

    files_hash = _sources_hash({n: all_files[n] for n in all_files if n.endswith((".h", ".cpp"))})
    prev_compile = (previous or {}).get("compile") or {}
    compile_state = (
        prev_compile if prev_compile.get("files_hash") == files_hash else {"status": "not_run"}
    )
    prev_verify = (previous or {}).get("verify") or {}
    verify_state = (
        prev_verify
        if prev_verify.get("files_hash") == verify.verify_hash(all_files)
        else {"status": "not_run"}
    )
    review = {
        "version": 1,
        "model": spec.model_id,
        "architecture": spec.hf_architecture,
        "slug": spec.slug,
        "out_dir": str(out_dir),
        "review_path": str(out_dir / "review.json"),
        "created_at": (previous or {}).get("created_at", _now()),
        "updated_at": _now(),
        "generator": generator,
        "attempts": attempts,
        "classes": {"transformer": spec.transformer_class, "causallm": spec.causallm_class},
        "files": {n: str(out_dir / n) for n in sorted(all_files)},
        "checklist": {
            "source": report.source,
            "fully_explained": report.is_fully_explained,
            "ticked": [
                {
                    "id": f.id,
                    "label": f.label,
                    "source": f.source,
                    "from": f.from_models,
                    "uncertain": f.uncertain,
                    "hooks": f.hooks,
                }
                for f in report.ticked
            ],
            "new_parts": plan["new_parts"],
        },
        "validation": vres.to_dict(),
        "parts": parts,
        "compile": compile_state,
        "verify": verify_state,
        "files_hash": files_hash,
        "other_config": spec.other_config,
        "summary": {
            "reuse": sum(1 for p in parts if p["status"] == "verified"),
            "to_review": sum(
                1
                for p in parts
                if p.get("needs_review") and p["status"] not in REVIEWED + ("rejected",)
            ),
            "checks_passed": vres.passed,
        },
    }
    _refresh_compile_ready(review)
    (out_dir / "review.json").write_text(
        json.dumps(review, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(_readme(review, spec), encoding="utf-8")
    return review


# Every upstream model source includes these. Copied (verified) hook bodies
# depend on them: withKey/createLayer come from llm_util.hpp, LayerHandle and
# the graph API from model.h. Found by verify: generated code without them
# does not build.
COMMON_INCLUDES = ("llm_util.hpp", "model.h", "app_context.h", "engine.h")


def ensure_common_includes(text: str, own_header: str) -> str:
    """Add the common model-source includes the file lacks, after its own header."""
    def has(hook: str) -> bool:
        return re.search(r'#\s*include\s*[<"]' + re.escape(hook) + r'[>"]', text) is not None
    missing = [h for h in COMMON_INCLUDES if not has(h)]
    if not missing:
        return text
    block = "".join(f"#include <{h}>\n" for h in missing)
    match = re.search(r'#\s*include\s*[<"]' + re.escape(own_header) + r'[>"][^\n]*\n', text)
    if match is None:
        match = re.search(r'^[ \t]*#\s*include[^\n]*\n', text, re.M)
        return text[:match.start()] + block + text[match.start():] if match else block + text
    return text[:match.end()] + block + text[match.end():]


def _readme(review: dict, spec: GenSpec) -> str:
    rows = "\n".join(
        f"| {p['hook']} | {p['action']} | {p['status']} |"
        f" {p.get('source_entry') or p.get('entry_id') or ''} |"
        for p in review["parts"]
    )
    return (
        f"# {spec.model_id} -- generated nntrainer CausalLM code\n\nGenerated by the Agentic AI"
        f" Workbench ({review['generator']}) on {review['updated_at']}.\nClasses:"
        f" `{spec.transformer_class}`, `{spec.causallm_class}` (registered as"
        f" `{spec.hf_architecture}` by `register_{spec.slug}_workbench.cpp`).\n\nChecks passed:"
        f" **{review['validation']['ok']}**. Review status lives in `review.json`; approve parts in"
        " the workbench's Review tab. Nothing here is reused for other models until it is approved"
        " and verified (build + logits match Hugging Face).\n\n| Part | Action | Status | Code"
        f" from |\n|---|---|---|---|\n{rows}\n\nBuild glue: `meson.build` adds the sources to"
        " `causallm_src`; `res/` holds the configs.\n"
    )


# ---------------------------------------------------------------- review
def load_review(path: os.PathLike) -> dict:
    """A review.json, with its compile and verify state filled in."""
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    review.setdefault("compile", {"status": "not_run"})  # reviews written before the compile step
    review.setdefault("verify", {"status": "not_run"})       # ... and before the verify step
    _refresh_compile_ready(review)
    return review


def _reload(
    review: dict,
) -> Tuple[GenSpec, ChecklistReport, dict, Path, Dict[str, str], dict, Optional[dict]]:
    out_dir = Path(review["out_dir"])
    work = out_dir / ".codegen"
    spec = GenSpec.from_dict(json.loads((work / "spec.json").read_text(encoding="utf-8")))
    data = json.loads((work / "checklist.json").read_text(encoding="utf-8"))
    report = _report_from_dict(data["report"])
    hf_sources = (
        json.loads((work / "hf_sources.json").read_text(encoding="utf-8"))
        if (work / "hf_sources.json").exists()
        else {}
    )
    raw = json.loads((out_dir / "res" / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((out_dir / "res" / "generation_config.json").read_text(encoding="utf-8"))
    return spec, report, data["plan"], out_dir, hf_sources, raw, gen


def _report_from_dict(data: dict) -> ChecklistReport:
    data = dict(data)
    data["features"] = [FeatureResult(**f) for f in data["features"]]
    return ChecklistReport(**data)


def _current_files(out_dir: Path, spec: GenSpec, review: dict) -> Dict[str, str]:
    names = [spec.header_name, spec.source_name] + [
        p["file"] for p in review["parts"] if p["kind"] == "layer"
    ]
    if spec.converter_action == "write":
        names.append("weight_converter.py")
    if spec.whole_model:
        names = [
            n
            for n in review["files"]
            if n.endswith((".h", ".cpp")) and not n.startswith("register_")
        ]
    return {n: (out_dir / n).read_text(encoding="utf-8") for n in names if (out_dir / n).exists()}


def recheck(review_path: os.PathLike, bank: CodeBank) -> dict:
    """Re-validate the files on disk (after the user edited them)."""
    review = load_review(review_path)
    spec, report, plan, out_dir, hf_sources, raw, gen = _reload(review)
    files = _current_files(out_dir, spec, review)
    vres = ValidationResult() if spec.whole_model else validate(files, spec, bank)
    register = None
    if spec.whole_model:
        register = dict(
            re.findall(
                r'registerModel\(\s*"([^"]+)".*?make_unique<(\w+)>',
                (out_dir / f"register_{spec.slug}_workbench.cpp").read_text(encoding="utf-8"),
                re.S,
            )
        )
    return _finalise(
        out_dir,
        spec,
        report,
        plan,
        files,
        vres,
        bank,
        hf_sources,
        raw,
        gen,
        register,
        source="edited",
        attempts=review.get("attempts", 0),
        generator=review.get("generator", ""),
        previous=review,
    )


# pylint: disable-next=too-many-locals,too-many-arguments
def regenerate(review_path: os.PathLike, part_id: str, comment: str, bank: CodeBank, llm: LLM,
               max_attempts: int = 3, log: Callable[[str], None] = print) -> dict:
    """Generate one part of a model again after a reviewer's comment."""
    review = load_review(review_path)
    spec, report, plan, out_dir, hf_sources, raw, gen = _reload(review)
    if spec.whole_model:
        raise CodegenError(
            "whole-model copies are not regenerated; edit the files and re-check instead"
        )
    work = out_dir / ".codegen"
    system = (work / "system_prompt.txt").read_text(encoding="utf-8")
    user = (work / "user_prompt.txt").read_text(encoding="utf-8")
    current = _current_files(out_dir, spec, review)
    if part_id == COMPILE_PART:
        heading, change = "Fix the compile errors", (
            "Fix exactly these errors (keep verified COPY code unchanged; "
            "fix around it, e.g. declare the members it uses)"
        )
    elif part_id == VERIFY_PART:
        heading, change = (
            "Fix the mismatch against Hugging Face",
            (
                "Find and fix the cause in the WRITE parts or the weight converter (keep verified"
                " COPY code unchanged)"
            ),
        )
    else:
        target = next((p for p in review["parts"] if p["id"] == part_id), None)
        if not target:
            raise CodegenError(f"no part {part_id!r}")
        heading = f"Reviewer feedback on {target['hook']}"
        change = f"Change {target['hook']} (and only what that change needs)"
    ask = (
        user
        + f"\n\n# {heading}\n{comment.strip() or '(no comment)'}\n{change}; keep everything else as"
        " in the current files. Return all files again in the same format.\n\n# Current files\n"
        + "\n".join(f"<<<FILE: {n}>>>\n{t}\n<<<END FILE>>>" for n, t in current.items())
    )
    files, vres, attempts = _llm_loop(spec, bank, llm, system, ask, work, max_attempts, log)
    return _finalise(
        out_dir,
        spec,
        report,
        plan,
        files or current,
        vres,
        bank,
        hf_sources,
        raw,
        gen,
        None,
        source="generated",
        attempts=review.get("attempts", 0) + attempts,
        generator=review.get("generator", "claude-cli"),
        previous=review,
    )


# pylint: disable-next=too-many-arguments
def set_part_status(review_path: os.PathLike, part_id: str, action: str, actor: str, bank: CodeBank,
                    reason: str = "") -> dict:
    """approve / reject a reviewed part; approval saves its learned box."""
    review = load_review(review_path)
    part = next((p for p in review["parts"] if p["id"] == part_id), None)
    if not part:
        raise CodegenError(f"no part {part_id!r}")
    if action == "approve":
        if not review["validation"]["ok"]:
            raise CodegenError(
                "fix the failed checks first (edit the files and re-check, or regenerate)"
            )
        if part["action"] == "write":
            if not part.get("entry_id"):
                raise CodegenError("this part has no code to approve")
            entry = bank.get(part["entry_id"])
            if entry and entry["status"] in ("generated", "failed_verification"):
                bank.approve(part["entry_id"], actor=actor, notes=reason)
            if part.get("learned_feature"):
                bank.put_feature(part["learned_feature"], source_entry=part["entry_id"])
            part["status"] = "approved"
        else:
            part["status"] = "accepted"  # upstream code the reviewer accepts for this model
    elif action == "reject":
        if part["action"] == "write" and part.get("entry_id"):
            entry = bank.get(part["entry_id"])
            if entry and entry["status"] in ("generated", "approved", "failed_verification"):
                bank.reject(part["entry_id"], actor=actor, reason=reason)
        part["status"] = "rejected"
    else:
        raise CodegenError(f"unknown action {action!r}")
    part.setdefault("history", []).append(
        {"action": action, "actor": actor, "at": _now(), "note": reason}
    )
    review["updated_at"] = _now()
    refresh_review_summary(review)
    Path(review_path).write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")
    return review


def refresh_review_summary(review: dict) -> None:
    """Recount what is left to review and whether the code may be compiled."""
    review.setdefault("summary", {})
    review["summary"]["reuse"] = sum(1 for p in review["parts"] if p.get("status") == "verified")
    review["summary"]["to_review"] = sum(1 for p in review["parts"] if p.get("needs_review")
                                         and p.get("status") not in REVIEWED + ("rejected",))
    _refresh_compile_ready(review)


# --------------------------------------------------------------- compile
def _sources_hash(files: Dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(files):
        hasher.update(name.encode("utf-8") + b"\0" + files[name].encode("utf-8") + b"\0")
    return hasher.hexdigest()[:16]


def compile_readiness(review: dict) -> Tuple[bool, str]:
    """Generated code is compiled only once it passed its checks and nothing
    is waiting for review (or was rejected)."""
    if not review.get("validation", {}).get("ok"):
        return (
            False,
            (
                "the generated code failed its checks -- fix them first (edit + Re-check, or"
                " Regenerate)"
            ),
        )
    rejected = [p["hook"] for p in review["parts"] if p.get("status") == "rejected"]
    if rejected:
        return False, f"rejected: {', '.join(rejected)} -- regenerate it first"
    pending = [p["hook"] for p in review["parts"]
               if p.get("needs_review") and p.get("status") not in REVIEWED]
    if pending:
        return False, f"waiting for review: {', '.join(pending)}"
    return True, "ready to compile"


def _refresh_compile_ready(review: dict) -> None:
    passed, why = compile_readiness(review)
    review["compile_ready"] = {"ok": passed, "why": why}


DOCKER_COMPILE_HINT = (" Or compile in Docker instead: start Docker Desktop and run "
                       "'AI Compiler Workbench: Set Up Docker Build'.")


def compile_generated(review_path: os.PathLike, compiler: Optional[str] = None,
                      nntrainer_root: Optional[os.PathLike] = None,
                      log: Optional[Callable[[str], None]] = None) -> dict:
    """Compile check of the generated sources; the result is stored in
    review.json. In Docker when that is the build host (and its build is set
    up): with the build's own compiler and flags, so a computer without a
    C++ compiler of its own can still compile and then verify. Otherwise
    syntax-only with this computer's compiler, against the checkout's headers.
    An explicit compiler (argument or $WORKBENCH_CXX) always runs here."""
    review = load_review(review_path)
    passed, why = compile_readiness(review)
    if not passed:
        raise CodegenError(f"Not compiled: {why}")
    root = Path(
        nntrainer_root
        or os.environ.get("NNTRAINER_ROOT")
        or (workbench_dir() or Path(".")).parent.parent
    )
    sources = [
        Path(p)
        for n, p in sorted(review["files"].items())
        if n.endswith(".cpp") and Path(p).exists()
    ]
    explicit = compiler or os.environ.get("WORKBENCH_CXX") or None
    result = None if explicit else compile_in_docker(review, sources, root, log=log)
    if result is None:
        result = compile_check.check(sources, root, explicit)
        if result.get("status") == "no_compiler":
            result["log"] = result.get("log", "") + DOCKER_COMPILE_HINT
    result["files_hash"] = review.get("files_hash", "")
    review["compile"] = result
    review["updated_at"] = _now()
    Path(review_path).write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")
    return review


# pylint: disable-next=too-many-locals
def compile_in_docker(review: dict, sources: List[Path], root: Path, ex=None,
                      log: Optional[Callable[[str], None]] = None) -> Optional[dict]:
    """The compile check in the Docker build container, or None when Docker is
    not the build host or its build is not set up yet. Each source is
    compiled with the build's rule and the flags of a libcausallm model source
    (verify.compile_source) into a container-only folder."""
    ex = ex or verify.select_executor(root)
    if ex.kind != "docker" or not ex.ninja_file().is_file():
        return None
    ninja = verify.Ninja(ex.ninja_file())
    if not ninja.model_compile_statement():
        return None
    out_dir = Path(review["out_dir"])
    slug = review.get("slug") or out_dir.name
    base = {"at": _now(), "files": [p.name for p in sources], "compiler": "g++-13 (Docker)",
            "mode": "the Docker build"}
    host = getattr(ex, "host", None)
    if host is not None:  # first, so an image rebuild (a new Dockerfile) is shown, not silent
        passed, detail = host.ensure_container(log)
        if not passed:
            return dict(
                base,
                status="failed",
                ok=False,
                log=detail,
                errors=[
                    {
                        "file": "",
                        "line": 0,
                        "message": "the Docker build container did not start: " + detail[-300:],
                    }
                ],
            )
    headers = [
        Path(p) for n, p in review["files"].items() if n.endswith(".h") and Path(p).is_file()
    ]
    passed, detail = ex.push([*sources, *headers])  # into the container's copy of the checkout
    if not passed:
        return dict(
            base,
            status="failed",
            ok=False,
            log=detail,
            errors=[
                {
                    "file": "",
                    "line": 0,
                    "message": "copying the generated code into Docker failed: " + detail[-300:],
                }
            ],
        )
    bdir = ex.build_dir
    objs = bdir / "wb_compile_check" / slug
    ex.mkdirs([objs])
    errors: List[dict] = []
    logs: List[str] = []
    for src in sources:
        _ok, errs, text = verify.compile_source(
            ninja, bdir, src, objs / (src.stem + ".o"), [out_dir], ex, 600
        )
        errors += errs
        logs.append(text)
    return dict(base, status="ok" if not errors else "failed", ok=not errors, errors=errors[:40],
                log="\n\n".join(logs)[-20000:])


def fix_compile_errors(review_path: os.PathLike, bank: CodeBank, llm: LLM, max_attempts: int = 3,
                       log: Callable[[str], None] = print) -> dict:
    """Send the compiler's errors back to Claude and regenerate."""
    review = load_review(review_path)
    errs = (review.get("compile") or {}).get("errors") or []
    if not errs:
        raise CodegenError("there are no compile errors to fix")
    text = "\n".join(f"{e['file']}:{e['line']}: {e['message']}" for e in errs[:30])
    return regenerate(
        review_path,
        COMPILE_PART,
        "The generated code does not compile. Compiler errors:\n" + text,
        bank,
        llm,
        max_attempts=max_attempts,
        log=log,
    )


def fix_verify_errors(review_path: os.PathLike, bank: CodeBank, llm: LLM, max_attempts: int = 3,
                      log: Callable[[str], None] = print) -> dict:
    """Send a failed verification (build errors, or the logit/token mismatch
    against Hugging Face) back to Claude and regenerate. The new code needs
    review again before it is compiled and verified."""
    review = load_review(review_path)
    try:
        comment = verify.verify_feedback(review)
    except verify.VerifyError as exc:
        raise CodegenError(str(exc)) from exc
    return regenerate(
        review_path, VERIFY_PART, comment, bank, llm, max_attempts=max_attempts, log=log
    )


COMPILE_PART = "__compile__"
VERIFY_PART = "__verify__"
# Part states that count as reviewed: approved/accepted by the user, or
# verified against Hugging Face.
REVIEWED = ("approved", "accepted", "verified")
