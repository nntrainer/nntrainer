# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file checklist.py
# @brief Feature checklist and coverage check for a Hugging Face causal LM.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Feature checklist and coverage check for a Hugging Face causal LM.

  facts = facts_from_dir("path/to/snapshot")        # config + tensor names/shapes
  report = run_checklist(facts, bank.learned_features())
  plan = match_bank(report, bank)                    # what can be reused

1. Every box (builtin_features.py, plus boxes the code bank learned) is
   checked against config.json and the checkpoint's tensor names/shapes.
2. Coverage: every tensor and every config key must be explained by a
   ticked box (or be metadata, or have a neutral value such as null/false).
   Whatever is left is a part the workbench has not seen before.
3. Planning: for each hook, the ticked features are matched against the code
   bank. Only *verified* entries are reused; approved-but-unverified matches
   are reported as needing verification; anything else is new code.

Reading a checkpoint never loads weights: only config.json and the
safetensors headers (or the index file) are read.

Rule language (each box's "when" list must all hold):
  {"tensor": glob}            some tensor matches
  {"no_tensor": glob}         no tensor matches
  {"shape": glob, "dims": [..]}  every matching tensor with a known shape has
                              these trailing dims (values may be "@derived")
  {"config": key, "op": op, "value": v}
        ops: exists, absent, null_or_absent, not_null, truthy, falsy,
             eq, ne, in, not_in, gt, lt, contains, all_in
        `key` may be a dotted path; "@name" keys/values are derived facts
        (head_dim, q_width, kv_width, hidden_size, rope_types, rope,
        rope_by_layer_kind, layer_kinds, tie_word_embeddings).
  {"any": [rules]}, {"all": [rules]}, {"not": rule}
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import argparse

from .builtin_features import BUILTIN_FEATURES, META_CONFIG_KEYS
from .code_bank import CodeBank

STRUCTURAL_HOOKS = (
    "constructModel",
    "createTransformerDecoderBlock",
    "createAttention",
    "createMlp",
)
COMPOSED_HOOKS = ("setupParameters", "registerCustomLayers", "converter")

_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")


# ------------------------------------------------------------------ facts
def normalise_tensor_name(name: str) -> str:
    """A tensor name without the model prefix, with layer numbers replaced by *."""
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "*", name)


def glob_to_regex(glob: str) -> re.Pattern:
    """A compiled regular expression for a checklist glob pattern."""
    out, i = [], 0
    while i < len(glob):
        char = glob[i]
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if char == "*":
            out.append(r"(?:\*|\d+)")
        elif char == "{":
            j = glob.index("}", i)
            out.append("(?:" + "|".join(re.escape(a) for a in glob[i + 1:j].split(",")) + ")")
            i = j
        else:
            out.append(re.escape(char))
        i += 1
    return re.compile("^" + "".join(out) + "$")


@dataclass
class ModelFacts:
    """What the checklist knows about a model: its config and its tensors."""
    name: str
    config: Dict[str, Any]                       # text config, resolved
    tensors: Dict[str, Optional[List[int]]]      # normalised name -> shape (None if unknown)
    tensor_examples: Dict[str, str] = field(default_factory=dict)   # normalised -> one raw name
    tensor_counts: Dict[str, int] = field(default_factory=dict)
    source: str = ""

    @property
    def tensors_known(self) -> bool:
        """True when the tensor list is known."""
        return bool(self.tensors)

    @property
    def shapes_known(self) -> bool:
        """True when at least one tensor shape is known."""
        return any(s is not None for s in self.tensors.values())


def resolve_text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Multimodal configs keep the language model under text_config."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config:
        merged = dict(text_config)
        for k in ("architectures", "model_type", "tie_word_embeddings"):
            if k in config and k not in merged:
                merged[k] = config[k]
        if "model_type" in text_config:
            merged["model_type"] = text_config["model_type"]
        merged.setdefault("architectures", config.get("architectures"))
        return merged
    return dict(config)


def make_facts(name: str, config: Dict[str, Any], raw_tensors: Dict[str, Optional[Sequence[int]]],
               source: str = "") -> ModelFacts:
    """ModelFacts from a config and the raw tensor names and shapes."""
    tensors: Dict[str, Optional[List[int]]] = {}
    examples: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for raw, shape in raw_tensors.items():
        tensor_name = normalise_tensor_name(raw)
        counts[tensor_name] = counts.get(tensor_name, 0) + 1
        examples.setdefault(tensor_name, raw)
        if tensor_name not in tensors or tensors[tensor_name] is None:
            tensors[tensor_name] = list(shape) if shape is not None else None
    return ModelFacts(name, resolve_text_config(config), tensors, examples, counts, source)


def read_safetensors_header(path: os.PathLike) -> Dict[str, Optional[List[int]]]:
    """Tensor names and shapes from a safetensors file's header (the weights are not read)."""
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"{path}: not a safetensors file")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    return {k: (v.get("shape") if isinstance(v, dict) else None) for k, v in header.items()}


def facts_from_dir(path: os.PathLike) -> ModelFacts:
    """config.json plus tensor names/shapes from the safetensors headers
    (names only from an index file if the shards are not there)."""
    data = Path(path)
    config = json.loads((data / "config.json").read_text(encoding="utf-8"))
    tensors: Dict[str, Optional[List[int]]] = {}
    shards = sorted(data.glob("*.safetensors"))
    for shard in shards:
        tensors.update(read_safetensors_header(shard))
    source = f"{len(shards)} safetensors header(s)" if shards else ""
    if not tensors:
        for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (data / idx).exists():
                weight_map = json.loads((data / idx).read_text(encoding="utf-8")).get(
                    "weight_map", {}
                )
                tensors = {k: None for k in weight_map}
                source = f"{idx} (names only, no shapes)"
                break
    return make_facts(data.name, config, tensors, source or "config only")


def facts_from_hub(
    repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
) -> ModelFacts:
    """Reads config.json and the safetensors headers over HTTP (range
    requests); nothing is downloaded beyond that. Needs huggingface_hub."""
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from huggingface_hub import get_safetensors_metadata, hf_hub_download  # optional dependency
    cfg_path = hf_hub_download(repo_id, "config.json", revision=revision, token=token)
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    meta = get_safetensors_metadata(repo_id, revision=revision, token=token)
    tensors: Dict[str, Optional[List[int]]] = {}
    for file_meta in meta.files_metadata.values():
        for tname, info in file_meta.tensors.items():
            tensors[tname] = list(info.shape)
    return make_facts(repo_id, config, tensors, "Hub safetensors headers")


def facts_from_transformers(config: Dict[str, Any], name: str = "") -> ModelFacts:
    """Tensor names/shapes from transformers' own model class, built on the
    meta device (no memory, no download). Needs torch + transformers."""
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    import torch  # optional dependency
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(**config) if "model_type" in config else None
    if cfg is None:
        raise ValueError("config has no model_type")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    state = model.state_dict()
    tied = getattr(cfg, "tie_word_embeddings", False)
    tensors = {k: list(v.shape) for k, v in state.items()
               if not (tied and k.endswith("lm_head.weight"))}
    return make_facts(
        name or config.get("model_type", "model"), config, tensors, "transformers (meta device)"
    )


# --------------------------------------------------------------- derived
def derived_facts(facts: ModelFacts) -> Dict[str, Any]:
    """Values computed from the config (head size, GQA, ...) that the rules can test."""
    cfg = facts.config
    heads = cfg.get("num_attention_heads") or 0
    hidden = cfg.get("hidden_size") or 0
    head_dim = cfg.get("head_dim") or (hidden // heads if heads else 0)
    num_key_value_heads = cfg.get("num_key_value_heads") or heads
    blocks: List[dict] = []
    by_kind: Dict[str, dict] = {}
    for src in (cfg.get("rope_scaling"), cfg.get("rope_parameters")):
        if isinstance(src, dict) and src:
            if all(isinstance(v, dict) for v in src.values()):
                by_kind.update(src)
                blocks.extend(src.values())
            else:
                blocks.append(src)
    rope_types = sorted(
        {str(b.get("rope_type") or b.get("type") or "default") for b in blocks}
    ) or ["default"]
    rope: Dict[str, Any] = {}
    for block in blocks:
        rope.update(block)
    layer_types = cfg.get("layer_types")
    return {
        "head_dim": head_dim,
        "hidden_size": hidden,
        "num_heads": heads,
        "num_kv_heads": num_key_value_heads,
        "q_width": heads * head_dim,
        "kv_width": num_key_value_heads * head_dim,
        "rope_types": rope_types,
        "rope": rope,
        "rope_by_layer_kind": by_kind,
        "layer_kinds": sorted(set(layer_types)) if isinstance(layer_types, list) else [],
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
    }


_MISSING = object()


def _lookup(facts: ModelFacts, derived: Dict[str, Any], key: str):
    if key.startswith("@"):
        cur: Any = derived
        parts = key[1:].split(".")
    else:
        cur = facts.config
        parts = key.split(".")
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _value(derived: Dict[str, Any], val):
    if isinstance(val, str) and val.startswith("@"):
        return derived.get(val[1:])
    return val


# ---------------------------------------------------------------- rules
@dataclass
class _Eval:
    passed: bool
    evidence: List[str]
    uncertain: bool = False


def _same(lhs, rhs) -> bool:
    """Equality that keeps True/1 and False/0 apart."""
    if isinstance(lhs, bool) or isinstance(rhs, bool) or lhs is None or rhs is None:
        return lhs is rhs
    return lhs == rhs


def _tensors_matching(facts: ModelFacts, glob: str) -> List[str]:
    regex = glob_to_regex(glob)
    return [t for t in facts.tensors if regex.match(t)]


# pylint: disable-next=R0914,R0912,R0915,R0911
def _eval_rule(rule: dict, facts: ModelFacts, derived: Dict[str, Any]) -> _Eval:
    if "any" in rule:
        results = [_eval_rule(r, facts, derived) for r in rule["any"]]
        hits = [r for r in results if r.passed]
        return _Eval(
            bool(hits),
            (
                hits[0].evidence
                if hits
                else ["none of: " + "; ".join(e for r in results for e in r.evidence)]
            ),
            any(r.uncertain for r in hits),
        )
    if "all" in rule:
        results = [_eval_rule(r, facts, derived) for r in rule["all"]]
        return _Eval(all(r.passed for r in results), [e for r in results for e in r.evidence],
                     any(r.uncertain for r in results))
    if "not" in rule:
        res = _eval_rule(rule["not"], facts, derived)
        return _Eval(not res.passed, ["not (" + "; ".join(res.evidence) + ")"], res.uncertain)
    if "tensor" in rule:
        if not facts.tensors_known:
            return _Eval(False, [f"tensor list not available ({rule['tensor']})"])
        tensors_matching = _tensors_matching(facts, rule["tensor"])
        return _Eval(
            bool(tensors_matching),
            [
                (
                    f"tensor {tensors_matching[0]}"
                    if tensors_matching
                    else f"no tensor {rule['tensor']}"
                )
            ],
        )
    if "no_tensor" in rule:
        if not facts.tensors_known:
            return _Eval(False, [f"tensor list not available ({rule['no_tensor']})"])
        tensors_matching = _tensors_matching(facts, rule["no_tensor"])
        return _Eval(
            not tensors_matching,
            [
                (
                    f"unexpected tensor {tensors_matching[0]}"
                    if tensors_matching
                    else f"no {rule['no_tensor']}"
                )
            ],
        )
    if "shape" in rule:
        want = [_value(derived, v) for v in rule["dims"]]
        tensors_matching = _tensors_matching(facts, rule["shape"])
        known = [(t, facts.tensors[t]) for t in tensors_matching if facts.tensors[t] is not None]
        if tensors_matching and not known:
            return _Eval(True, [f"shape of {rule['shape']} not available"], uncertain=True)
        bad = [(t, s) for t, s in known if list(s[-len(want):]) != want]
        if bad:
            return _Eval(
                False,
                [f"{bad[0][0]} has shape {bad[0][1]}, expected [..., {', '.join(map(str, want))}]"],
            )
        return _Eval(True, [f"{t} shape {s}" for t, s in known[:1]])
    if "config" in rule:
        key, oper = rule["config"], rule.get("op", "exists")
        got = _lookup(facts, derived, key)
        val = _value(derived, rule.get("value"))
        present = got is not _MISSING
        found = None if not present else got
        if oper == "exists":
            passed = present
        elif oper == "absent":
            passed = not present
        elif oper == "null_or_absent":
            passed = found is None
        elif oper == "not_null":
            passed = found is not None
        elif oper == "truthy":
            passed = bool(found)
        elif oper == "falsy":
            passed = not found
        elif oper == "eq":
            passed = present and _same(found, val)
        elif oper == "ne":
            passed = not _same(found, val)
        elif oper == "in":
            passed = any(_same(found, v) for v in val)
        elif oper == "not_in":
            passed = not any(_same(found, v) for v in val)
        elif oper == "gt":
            passed = isinstance(found, (int, float)) and not isinstance(found, bool) and found > val
        elif oper == "lt":
            passed = isinstance(found, (int, float)) and not isinstance(found, bool) and found < val
        elif oper == "contains":
            passed = isinstance(found, (list, tuple)) and val in found
        elif oper == "all_in":
            passed = (
                isinstance(found, (list, tuple)) and all(x in val for x in found)
                if present
                else True
            )
        else:
            raise ValueError(f"unknown op {oper}")
        shown = "absent" if not present else json.dumps(found)[:80]
        return _Eval(
            bool(passed),
            [
                f"{key} = {shown}"
                + ("" if oper in ("exists", "absent") else f" ({oper} {json.dumps(val)[:60]})")
            ],
        )
    raise ValueError(f"unknown rule {rule}")


# --------------------------------------------------------------- report
@dataclass
# pylint: disable-next=too-many-instance-attributes
class FeatureResult:
    """One checklist feature, whether the model has it, and the evidence."""
    id: str  # pylint: disable=invalid-name  # the key the webview's checklist reads
    label: str
    hooks: List[str]
    ticked: bool
    evidence: List[str]
    uncertain: bool = False
    source: str = "builtin"
    from_models: List[str] = field(default_factory=list)
    level: str = "feature"
    model_entry: str = ""
    note: str = ""
    claimed_tensors: List[str] = field(default_factory=list)
    claimed_config: List[str] = field(default_factory=list)


@dataclass
# pylint: disable-next=too-many-instance-attributes
class ChecklistReport:
    """The checklist result: the features, and what no feature explains."""
    model: str
    architectures: List[str]
    model_type: str
    source: str
    tensors_known: bool
    shapes_known: bool
    features: List[FeatureResult]
    unclaimed_tensors: List[Dict[str, Any]]
    unclaimed_config: Dict[str, Any]

    @property
    def ticked(self) -> List[FeatureResult]:
        """The features the model has."""
        return [f for f in self.features if f.ticked]

    def ticked_for(self, hook: str) -> List[str]:
        """Ids of the ticked features that belong to `hook`."""
        return sorted(f.id for f in self.ticked if hook in f.hooks)

    @property
    def is_fully_explained(self) -> bool:
        """True when every tensor and config key is claimed by a feature."""
        return self.tensors_known and not self.unclaimed_tensors and not self.unclaimed_config

    def to_dict(self) -> Dict[str, Any]:
        """The report as JSON-ready data."""
        return asdict(self)


def _is_neutral(val: Any) -> bool:
    return val is None or val is False or val == [] or val == {} or val == ""


# pylint: disable-next=too-many-locals
def run_checklist(facts: ModelFacts, extra_features: Iterable[dict] = ()) -> ChecklistReport:
    """Tick the built-in (and learned) features against the model's facts."""
    derived = derived_facts(facts)
    specs = list(BUILTIN_FEATURES) + [
        dict(f, source=f.get("source", "learned")) for f in extra_features
    ]
    results: List[FeatureResult] = []
    claimed_t: set = set()
    claimed_c: set = set()
    for spec in specs:
        evals = [_eval_rule(r, facts, derived) for r in spec["when"]]
        ticked = all(e.passed for e in evals)
        evidence = [x for e in evals for x in e.evidence] if ticked else \
                   [x for e in evals if not e.passed for x in e.evidence]
        result = FeatureResult(
            id=spec["id"], label=spec.get("label", spec["id"]), hooks=list(spec.get("hooks", [])),
            ticked=ticked, evidence=evidence, uncertain=ticked and any(e.uncertain for e in evals),
            source=spec.get("source", "builtin"), from_models=list(spec.get("from", [])),
            level=spec.get("level", "feature"), model_entry=spec.get("model_entry", ""),
            note=spec.get("note", ""))
        if ticked:
            claims = spec.get("claims", {})
            excl = [glob_to_regex(g) for g in spec.get("claims_exclude", [])]
            for tensor in claims.get("tensors", []):
                for tensors_matching_item in _tensors_matching(facts, tensor):
                    if not any(x.match(tensors_matching_item) for x in excl):
                        result.claimed_tensors.append(tensors_matching_item)
            ckeys = claims.get("config", [])
            result.claimed_config = (
                sorted(facts.config) if "*" in ckeys else [k for k in ckeys if k in facts.config]
            )
            # A learned box exists to explain what the built-in boxes could
            # not (e.g. OLMo-2's full-width q_norm). If everything it claims
            # is already explained (Qwen3's per-head q_norm), it does not
            # apply -- otherwise it would pull verified models off their code.
            if (
                result.source == "learned"
                and result.claimed_tensors + result.claimed_config
                and set(result.claimed_tensors) <= claimed_t
                and set(result.claimed_config) <= claimed_c
            ):
                result.ticked = False
                result.evidence = ["everything it describes is already explained by built-in boxes"]
                result.claimed_tensors, result.claimed_config = [], []
            claimed_t.update(result.claimed_tensors)
            claimed_c.update(result.claimed_config)
        results.append(result)
    unclaimed_t = [
        {
            "pattern": t,
            "count": facts.tensor_counts.get(t, 1),
            "example": facts.tensor_examples.get(t, t),
            "shape": facts.tensors[t],
            "hook": guess_hook(t),
        }
        for t in sorted(facts.tensors)
        if t not in claimed_t
    ]
    unclaimed_c = {k: v for k, v in sorted(facts.config.items())
                   if k not in claimed_c and k not in META_CONFIG_KEYS and not _is_neutral(v)}
    return ChecklistReport(
        model=facts.name, architectures=list(facts.config.get("architectures") or []),
        model_type=str(facts.config.get("model_type", "")), source=facts.source,
        tensors_known=facts.tensors_known, shapes_known=facts.shapes_known,
        features=results, unclaimed_tensors=unclaimed_t, unclaimed_config=unclaimed_c)


def guess_hook(name: str) -> str:
    """Where an unexplained tensor or config key most likely needs code."""
    lowered = name.lower()
    # A norm that sits directly in a decoder layer (input_layernorm,
    # post_attention_layernorm, ...) belongs to the block, whatever its name says.
    if re.match(r"^layers\.\*\.[a-z_]*(layernorm|_norm|norm)\.weight$", lowered):
        return "createTransformerDecoderBlock"
    if re.search(
        r"attn|attention|rope|rotary|sliding|window|head_dim|q_|k_|qkv|query|sink|softcap", lowered
    ):
        return "createAttention"
    if re.search(
        r"mlp|ffn|feed_forward|expert|moe|router|gate|up_proj|down_proj|act|intermediate", lowered
    ):
        return "createMlp"
    if re.search(r"layers\.\*\..*norm|layernorm", lowered):
        return "createTransformerDecoderBlock"
    if re.search(r"embed|lm_head|^norm|vocab|logit", lowered):
        return "constructModel"
    return "setupParameters"


# --------------------------------------------------------------- planning
# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def match_bank(report: ChecklistReport, bank) -> Dict[str, Any]:
    """Per hook: reuse a verified entry, flag a match that is not verified
    yet, or mark the hook as needing new code."""
    plan: Dict[str, Any] = {"model": report.model, "whole_model": None, "hooks": {}, "composed": {},
                            "converter": None, "new_parts": [], "summary": {}}
    new_by_hook: Dict[str, List[str]] = {}
    for unclaimed_tensor in report.unclaimed_tensors:
        new_by_hook.setdefault(unclaimed_tensor["hook"], []).append(
            f"tensor {unclaimed_tensor['pattern']} (x{unclaimed_tensor['count']})"
        )
    for k, val in report.unclaimed_config.items():
        new_by_hook.setdefault(guess_hook(k), []).append(f"config {k} = {json.dumps(val)[:80]}")
    for feature in report.features:
        if feature.ticked and feature.uncertain:
            plan["new_parts"].append(
                {
                    "hook": ", ".join(feature.hooks),
                    "what": f"check {feature.id}: " + "; ".join(feature.evidence),
                }
            )

    whole = [f for f in report.ticked if f.level == "model"]
    if whole:
        feature = whole[0]
        entry = bank.get(feature.model_entry) if feature.model_entry else None
        plan["whole_model"] = {"feature": feature.id, "entry": feature.model_entry,
                               "status": entry["status"] if entry else "missing",
                               "state": _state_for(entry["status"] if entry else None)}

    for hook in STRUCTURAL_HOOKS:
        ticked = report.ticked_for(hook)
        info: Dict[str, Any] = {
            "features": ticked,
            "reuse": None,
            "unverified_matches": [],
            "examples": [],
            "new": new_by_hook.get(hook, []),
        }
        whole = plan.get("whole_model")
        if whole:
            info["state"] = whole["state"]
            info["reuse"] = whole["entry"] if info["state"] == "reuse" else None
            plan["hooks"][hook] = info
            continue
        reusable = bank.find_reusable("hook", ticked, hook=hook) if ticked else []
        approved = (
            list(bank.find_matching("hook", ticked, hook=hook, statuses=("approved",)))
            if ticked
            else []
        )
        if reusable:
            info["reuse"] = reusable[0]["id"]
            info["alternatives"] = [e["id"] for e in reusable[1:]]
        info["unverified_matches"] = [e["id"] for e in approved]
        if info["new"] or not ticked:
            info["state"] = "new"
        elif info["reuse"]:
            info["state"] = "reuse"
        elif approved:
            info["state"] = "needs_verification"
        else:
            info["state"] = "new"
        if info["state"] != "reuse":
            # Search on what is distinctive: the new modules / config keys
            # (q_norm, rope_scaling, ...), not words every tensor name has.
            distinct = [_distinct_name(n.split(" ")[1]) for n in info["new"]]
            query = " ".join(distinct or ticked)
            hits = bank.search(query, kind="hook", hook=hook, limit=6, embeddings="off")
            # Whole-model code (Gemma4, LFM2) is a poor starting point for a
            # single hook; prefer ordinary hooks (the sort is stable).
            hits.sort(key=lambda e: any(f.endswith("_model") for f in e["features"]))
            info["examples"] = [e["id"] for e in hits[:3]]
        plan["hooks"][hook] = info

    for hook in COMPOSED_HOOKS:
        feats = report.ticked_for(hook)
        sources = sorted({m for f in report.ticked if hook in f.hooks for m in f.from_models})
        plan["composed"][hook] = {
            "features": feats,
            "taken_from": sources,
            "new": new_by_hook.get(hook, []) if hook == "setupParameters" else [],
        }

    # A weight converter only depends on the tensor layout.
    layout = [f.id for f in report.ticked if f.claimed_tensors or "converter" in f.hooks]
    conv = bank.find_reusable("converter", layout)
    conv_unverified = bank.find_matching("converter", layout, statuses=("approved",))
    plan["converter"] = {
        "reuse": conv[0]["id"] if conv else None,
        "unverified_matches": [e["id"] for e in conv_unverified],
        "state": "new" if report.unclaimed_tensors else ("reuse" if conv else
                 "needs_verification" if conv_unverified else "new"),
    }

    for hook, items in new_by_hook.items():
        for what in items:
            plan["new_parts"].append({"hook": hook, "what": what})
    states = [h["state"] for h in plan["hooks"].values()]
    plan["summary"] = {s: states.count(s) for s in ("reuse", "needs_verification", "new")}
    plan["summary"]["fully_explained"] = report.is_fully_explained
    return plan


def _distinct_name(name: str) -> str:
    """layers.*.self_attn.q_norm.weight -> q_norm ; rope_scaling -> rope_scaling"""
    parts = [p for p in name.split(".") if p not in ("*", "weight", "bias", "layers")]
    return parts[-1] if parts else name


def _state_for(status: Optional[str]) -> str:
    if status == "verified":
        return "reuse"
    if status == "approved":
        return "needs_verification"
    return "new"


# -------------------------------------------------------------------- CLI
_STATE_LABEL = {
    "reuse": "reuse (verified)",
    "needs_verification": "match, NOT verified -> verify first",
    "new": "NEW code needed",
}


# pylint: disable-next=too-many-locals,too-many-branches
def format_report(report: ChecklistReport, plan: Optional[Dict[str, Any]] = None) -> str:
    """The report, and the plan when given, as readable text."""
    lines = [f"Checklist: {report.model}  ({', '.join(report.architectures) or report.model_type})",
             f"Read from: {report.source}"]
    if not report.tensors_known:
        lines.append("!! No tensor list: tensor-based boxes cannot tick. Point at a snapshot with "
                     "*.safetensors, use --hub, or --via-transformers.")
    lines.append("")
    for feature in report.features:
        if not feature.ticked:
            continue
        mark = "[?]" if feature.uncertain else "[x]"
        src = (
            "learned"
            if feature.source == "learned"
            else "built-in (" + ", ".join(feature.from_models) + ")"
        )
        lines.append(f" {mark} {feature.label:<62} {src}")
        if feature.uncertain:
            lines.append(f"      unconfirmed: {'; '.join(feature.evidence)}")
    if report.unclaimed_tensors or report.unclaimed_config:
        lines.append("\nNot explained by any box (new parts):")
        for unclaimed_tensor in report.unclaimed_tensors:
            lines.append(
                f" [!] tensor {unclaimed_tensor['pattern']}  x{unclaimed_tensor['count']}  shape"
                f" {unclaimed_tensor['shape']}  -> {unclaimed_tensor['hook']}"
            )
        for k, val in report.unclaimed_config.items():
            lines.append(f" [!] config {k} = {json.dumps(val)[:100]}  -> {guess_hook(k)}")
    else:
        lines.append("\nEvery tensor and config key is explained by a ticked box.")
    if plan:
        lines.append("\nPlan against the code bank:")
        if plan["whole_model"]:
            whole_model = plan["whole_model"]
            lines.append(
                f"  whole model -> {whole_model['entry']} ({whole_model['status']}):"
                f" {_STATE_LABEL[whole_model['state']]}"
            )
        def target(hdr: Dict[str, Any]) -> str:
            if hdr["state"] == "reuse":
                return hdr["reuse"] or ""
            if hdr["state"] == "needs_verification":
                return hdr["unverified_matches"][0] if hdr["unverified_matches"] else ""
            return ""
        for hook, hook_plan in plan["hooks"].items():
            lines.append(f"  {hook:<30} {_STATE_LABEL[hook_plan['state']]:<38} {target(hook_plan)}")
            for what in hook_plan.get("new", []):
                lines.append(f"  {'':<30} because: {what}")
            if hook_plan["state"] != "reuse" and hook_plan.get("examples"):
                lines.append(f"  {'':<30} start from: {', '.join(hook_plan['examples'])}")
        converter = plan["converter"]
        lines.append(
            f"  {'weight converter':<30} {_STATE_LABEL[converter['state']]:<38} {target(converter)}"
        )
        for hook, comp in plan["composed"].items():
            if comp["features"]:
                lines.append(f"  {hook:<30} composed from: {', '.join(comp['taken_from'])}")
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge.checklist",
        description="Tick the feature boxes for a HF causal LM and plan against the code bank.",
    )
    parser.add_argument(
        "model", help="local snapshot folder, a config.json, or a Hub repo id with --hub"
    )
    parser.add_argument(
        "--hub", action="store_true", help="read config + safetensors headers from the Hub"
    )
    parser.add_argument(
        "--via-transformers",
        action="store_true",
        help="get tensor names/shapes from transformers on the meta device (needs torch)",
    )
    parser.add_argument(
        "--db", help="database path (default: tools/agentic-ai-workbench/code_bank/code_bank.db)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.hub:
        facts = facts_from_hub(args.model)
    else:
        path = Path(args.model)
        if args.via_transformers:
            cfg_file = path if path.is_file() else path / "config.json"
            facts = facts_from_transformers(json.loads(cfg_file.read_text(encoding="utf-8")),
                                            cfg_file.parent.name)
        elif path.is_file():
            facts = make_facts(
                path.parent.name, json.loads(path.read_text(encoding="utf-8")), {}, "config only"
            )
        else:
            facts = facts_from_dir(path)
    with CodeBank(args.db) as bank:
        report = run_checklist(facts, bank.learned_features())
        plan = match_bank(report, bank)
    if args.json:
        print(json.dumps({"report": report.to_dict(), "plan": plan}, indent=2))
    else:
        print(format_report(report, plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
