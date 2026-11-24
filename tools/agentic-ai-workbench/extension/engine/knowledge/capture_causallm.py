# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file capture_causallm.py
# @brief Fill the code bank from nntrainer's hand-written CausalLM code (Applications/CausalLM).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Fill the code bank from nntrainer's hand-written CausalLM code
(Applications/CausalLM).

  python -m knowledge.capture_causallm [--nntrainer-root PATH] [--db PATH]

Captured, so generation still works if models/ is not there:
  framework  models/transformer, causal_lm, sentence_transformer, ...
  model      every models/<arch>/ folder, as written
  hook       every method each model class defines, tagged with the
             checklist features it implements (structural hooks)
  layer      custom layers in models/<arch>/ and the shared layers/
  converter  res/**/*.py and the reference-fixture generators

What counts as verified (the reuse rule): an entry is `verified` only if an
upstream reference test exercises it -- test/unittest/models/
unittest_causallm_*_reference.cpp, which compares nntrainer's output with HF
transformers logits (fixture meta.json: logits_atol_fp32). Which methods a
test exercises is worked out from the class hierarchy (the C++ final
overrider of each method, plus base methods the overriders call explicitly).
Everything else hand-written is `approved` (merged upstream, but no
reference test covers it): usable as an example, never reused automatically.
The workbench does not re-run those tests; the evidence records that.

Re-running is safe: unchanged entries are left alone, changed ones updated.
Standard library only.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse

from .builtin_features import BUILTIN_FEATURES
from .code_bank import CodeBank
from .cpp_extract import (CppDefinition, calls_in, final_overriders, find_classes, find_definitions,
                          find_inline_methods, find_layer_types, layer_types_used, string_literals)

# ------------------------------------------------------------ annotations
_COMMON = ["embedding", "decoder_stack", "final_norm_rms"]
_LM_HEAD = ["lm_head_tied", "lm_head_untied"]

#: model folder -> (features the model implements, features it also handles)
MODEL_FEATURES: Dict[str, Tuple[List[str], List[str]]] = {
    "qwen2": (
        _COMMON + ["block_pre_norm", "attn_gqa_rope", "rope_default", "qkv_bias", "mlp_gated_silu"],
        _LM_HEAD + ["sliding_window_uniform"],
    ),
    "qwen3": (
        _COMMON
        + ["block_pre_norm", "attn_gqa_rope", "rope_default", "qk_norm_per_head", "mlp_gated_silu"],
        _LM_HEAD + ["sliding_window_uniform"],
    ),
    "qwen3_moe": (
        _COMMON
        + ["block_pre_norm", "attn_gqa_rope", "rope_default", "qk_norm_per_head", "moe_qwen"],
        _LM_HEAD + ["sliding_window_uniform"],
    ),
    "gpt_oss": (
        _COMMON
        + [
            "block_pre_norm",
            "attn_gqa_rope",
            "qkv_bias",
            "o_proj_bias",
            "attention_sinks",
            "rope_yarn",
            "sliding_window_layer_types",
            "moe_gpt_oss",
        ],
        _LM_HEAD,
    ),
    "gemma3": (
        _COMMON
        + [
            "block_sandwich_norm",
            "attn_gqa_rope",
            "rope_default",
            "qk_norm_per_head",
            "local_rope_theta_10000",
            "mlp_gated_gelu_tanh",
            "embedding_scale_sqrt_hidden",
            "norm_one_plus_weight",
        ],
        _LM_HEAD + ["sliding_window_pattern", "sliding_window_layer_types"],
    ),
    "gemma4": (["gemma4_model"], []),
    "lfm2": (["lfm2_model"], []),
}
MODEL_FEATURES["qwen3_slim_moe"] = MODEL_FEATURES["qwen3_cached_slim_moe"] = MODEL_FEATURES[
    "qwen3_moe"
]
MODEL_FEATURES["gpt_oss_cached_slim"] = MODEL_FEATURES["gpt_oss"]

#: Class::method -> (features it implements, features it also handles).
#: Only these structural hooks are matched against a new model's checklist.
HOOK_FEATURES: Dict[str, Tuple[List[str], List[str]]] = {
    "Transformer::constructModel": (_COMMON, []),
    "CausalLM::constructModel": (_COMMON, _LM_HEAD),
    "Transformer::createTransformerDecoderBlock": (["block_pre_norm"], []),
    "Transformer::createAttention": (["attn_gqa_rope", "rope_default"], ["sliding_window_pattern"]),
    "Transformer::createMlp": (["mlp_gated_silu"], []),
    "Qwen2Transformer::createAttention": (
        ["attn_gqa_rope", "rope_default", "qkv_bias"],
        ["sliding_window_uniform"],
    ),
    "Qwen3Transformer::createAttention": (
        ["attn_gqa_rope", "rope_default", "qk_norm_per_head"],
        ["sliding_window_uniform"],
    ),
    "Gemma3Transformer::createTransformerDecoderBlock": (["block_sandwich_norm"], []),
    # attn_logit_softcapping is passed to mha_core here, but its value is only
    # read in Gemma3Transformer::setupParameters, which never runs (the
    # constructors dispatch to the base setupParameters), so it is not listed.
    "Gemma3Transformer::createAttention": (
        ["attn_gqa_rope", "rope_default", "qk_norm_per_head", "local_rope_theta_10000"],
        ["sliding_window_uniform", "sliding_window_pattern", "sliding_window_layer_types"],
    ),
    "Gemma3Transformer::createMlp": (["mlp_gated_gelu_tanh"], []),
    "Qwen3MoECausalLM::createMlp": (["moe_qwen"], []),
    "Qwen3SlimMoECausalLM::createMlp": (["moe_qwen"], []),
    "Qwen3CachedSlimMoECausalLM::createMlp": (["moe_qwen"], []),
}
for _cls in ("GptOssForCausalLM", "GptOssCachedSlimCausalLM"):
    HOOK_FEATURES[f"{_cls}::createAttention"] = (
        [
            "attn_gqa_rope",
            "qkv_bias",
            "o_proj_bias",
            "attention_sinks",
            "rope_yarn",
            "sliding_window_layer_types",
        ],
        [],
    )
    HOOK_FEATURES[f"{_cls}::createMlp"] = (["moe_gpt_oss"], [])

HOOK_VARIANTS = {
    "Qwen3SlimMoECausalLM::createMlp": "slim",
    "Qwen3CachedSlimMoECausalLM::createMlp": "cached_slim",
    "GptOssCachedSlimCausalLM::createAttention": "cached_slim",
    "GptOssCachedSlimCausalLM::createMlp": "cached_slim",
}

STRUCTURAL = ("constructModel", "createTransformerDecoderBlock", "createAttention", "createMlp")
WHOLE_MODEL_PREFIX = {"Gemma4": "gemma4_model", "Lfm2": "lfm2_model"}

FRAMEWORK = {
    "framework:causallm-base": (
        "CausalLM base classes (Transformer, CausalLM)",
        [
            "models/transformer.h",
            "models/transformer.cpp",
            "models/causal_lm.h",
            "models/causal_lm.cpp",
            "models/performance_metrics.h",
        ],
        ["Transformer", "CausalLM"],
    ),
    "framework:sentence-transformer": (
        "SentenceTransformer base class",
        ["models/sentence_transformer.h", "models/sentence_transformer.cpp"],
        ["SentenceTransformer"],
    ),
    "framework:models-meson": (
        "models/meson.build (how model folders are built)",
        ["models/meson.build"],
        [],
    ),
}
FRAMEWORK_CLASSES = {"Transformer", "CausalLM", "SentenceTransformer"}

#: res/ converter path prefix -> model folder whose features it converts
CONVERTER_MODEL = {
    "qwen2/": "qwen2",
    "qwen3/qwen3-0.6b/": "qwen3",
    "qwen3/qwen3-4b/": "qwen3",
    "qwen3/qwen3-30b-a3b/": "qwen3_moe",
    "gpt-oss/": "gpt_oss",
    "gemma3/": "gemma3",
    "gemma4/": "gemma4",
}
GENERATOR_MODEL = {"qwen2": "qwen2", "qwen3": "qwen3", "qwen3_moe": "qwen3_moe", "gemma3": "gemma3",
                   "gemma4": "gemma4", "lfm2": "lfm2"}


#: Features that change the tensor layout (or the converter itself) -- the
#: only ones a weight converter depends on.
LAYOUT_FEATURES = {f["id"] for f in BUILTIN_FEATURES
                   if f.get("claims", {}).get("tensors") or "converter" in f.get("hooks", [])}


def _layout(features: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
    req, sup = features
    return [f for f in req if f in LAYOUT_FEATURES], [f for f in sup if f in LAYOUT_FEATURES]


class CaptureError(RuntimeError):
    """The CausalLM sources could not be captured."""


def find_nntrainer_root(start: Optional[Path] = None) -> Optional[Path]:
    """The nntrainer checkout above `start` (or this file), or None."""
    path = (start or Path(__file__)).resolve()
    for cand in [path, *path.parents]:
        if (cand / "Applications" / "CausalLM" / "models").is_dir():
            return cand
    return None


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                           encoding="utf-8", timeout=20, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ capture
# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def capture(nntrainer_root: Path, bank: CodeBank) -> Dict[str, Any]:
    """Fill the code bank from the checkout's Applications/CausalLM."""
    root = Path(nntrainer_root).resolve()
    cl_path = root / "Applications" / "CausalLM"
    models = cl_path / "models"
    if not models.is_dir():
        raise CaptureError(
            f"{models} not found -- point --nntrainer-root at the nntrainer checkout"
        )
    commit = _git(root, "rev-parse", "HEAD")
    commit_date = _git(root, "log", "-1", "--format=%cI")
    dirty = bool(_git(root, "status", "--porcelain", "--", "Applications/CausalLM"))
    ref = (commit[:12] or "unknown-commit") + ("+local-changes" if dirty else "")

    # -- 1. read the C++ ----------------------------------------------------
    src: Dict[str, str] = {}
    for sub in ("models", "layers"):
        for path in sorted((cl_path / sub).rglob("*")):
            if path.suffix in (".h", ".cpp", ".hpp") and path.is_file():
                src[path.relative_to(cl_path).as_posix()] = _read(path)
    bases: Dict[str, List[str]] = {}
    class_home: Dict[str, str] = {}
    defs: Dict[str, Dict[str, List[Tuple[str, CppDefinition]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    file_types: Dict[str, List[str]] = {}
    for rel, text in src.items():
        for find_class in find_classes(text):
            bases.setdefault(find_class.name, find_class.bases)
            if rel.endswith(".h") or find_class.name not in class_home:
                class_home[find_class.name] = rel
        types = find_layer_types(text)
        if types:
            file_types[rel] = types
        found = find_definitions(text) if rel.endswith(".cpp") else []
        found += find_inline_methods(text)
        for found_item in found:
            defs[found_item.class_name][found_item.method].append((rel, found_item))
    layer_stems = {_stem_key(rel) for rel in file_types} | {
        _stem_key(r) for r in src if r.startswith("layers/")
    }
    defined = {c: set(m) for c, m in defs.items()}
    custom_types = {t for v in file_types.values() for t in v}

    # -- 2. registrations and reference tests ---------------------------------
    registered = _registrations(cl_path / "main.cpp")
    tests_dir = root / "test" / "unittest" / "models"
    tested, fixtures, test_calls = _reference_tests(tests_dir, set(bases))
    atol = _fixture_atol(tests_dir / "causallm_reference", fixtures)

    # -- 3. what the reference tests exercise --------------------------------
    # A method counts as tested only if it is reachable from what runs in the
    # test: every constructor in the class hierarchy (inside a constructor,
    # virtual calls resolve to that constructor's class, as in C++) and the
    # methods the test calls (resolved to their final overrider).
    exercised: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    final_by_class: Dict[str, Dict[str, str]] = {}
    for cls, tests in tested.items():
        final_by_class[cls] = final_overriders(cls, bases, defined)
        call_names = set().union(*[test_calls.get(t, set()) for t in tests])
        for owner, meth in _reachable(cls, bases, defined, defs, call_names):
            exercised[(owner, meth)] |= tests
    type_tests: Dict[str, Set[str]] = defaultdict(set)      # layer type -> tests that create it
    type_users: Dict[str, Set[str]] = defaultdict(set)      # layer type -> exercised Class::method
    for (owner, meth), tests in exercised.items():
        for rel, found_item in defs.get(owner, {}).get(meth, []):
            # createLayer("type", ...) and types picked through a variable
            # (e.g. TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head").
            for layer_type in set(layer_types_used(found_item.body)) | (
                set(string_literals(found_item.body)) & custom_types
            ):
                type_tests[layer_type] |= tests
                type_users[layer_type].add(f"{owner}::{meth}")
    # Layer types named in a file-scope table (e.g. SentenceTransformer's
    # {"Pooling", "embedding_pooling"} module map) count for the tests that
    # exercise that file's methods.
    file_tests: Dict[str, Set[str]] = defaultdict(set)
    for (owner, meth), tests in exercised.items():
        for rel, _d in defs.get(owner, {}).get(meth, []):
            file_tests[rel] |= tests
    for rel, tests in file_tests.items():
        text = src[rel]
        for _rel2, found_item in [
            x for c in defs.values() for v in c.values() for x in v if x[0] == rel
        ]:
            text = text.replace(found_item.body, "")
        for layer_type in set(string_literals(text)) & custom_types:
            type_tests[layer_type] |= tests
            type_users[layer_type].add(f"{rel} (file-scope table)")

    def verification(tests: Set[str]) -> Dict[str, Any]:
        fixture_names = sorted({f for t in tests for f in fixtures.get(t, [])})
        return {
            "method": "upstream-reference-tests",
            "tests": sorted(tests),
            "fixtures": fixture_names,
            "logits_atol_fp32": atol,
            "commit": ref,
            "note": (
                "Hand-written nntrainer code exercised by reference tests that compare its output "
                "with HF transformers logits. The workbench did not re-run these tests."
            ),
        }

    upstream = {"approved_by": "nntrainer upstream (merged code)", "approved_at": commit_date}
    counts: Dict[str, int] = defaultdict(int)
    not_verified: List[str] = []

    def put(entry: Dict[str, Any], tests: Set[str]) -> None:
        entry.setdefault("origin", "models")
        entry.update(source="captured", **upstream)
        entry.setdefault("source_ref", ref)
        if tests:
            entry.update(status="verified", verification=verification(tests))
        else:
            entry.update(status="approved", verification={})
            not_verified.append(entry["id"])
        counts[bank.put(entry, actor="capture")] += 1

    # -- 4. framework ----------------------------------------------------------
    for fid, (name, files, fclasses) in FRAMEWORK.items():
        present = {f: _read(cl_path / f) for f in files if (cl_path / f).exists()}
        if not present:
            continue
        tests = (
            set().union(*[t for (c, _m), t in exercised.items() if c in fclasses])
            if fclasses
            else set()
        )
        put({"id": fid, "kind": "framework", "name": name, "architecture": "base", "files": present,
             "evidence": {"classes": fclasses, "emit_when_models_missing": True}}, tests)

    # -- 5. model folders ------------------------------------------------------
    for found_item in sorted(p for p in models.iterdir() if p.is_dir()):
        arch = found_item.name
        files = {
            p.relative_to(cl_path).as_posix(): _read(p)
            for p in sorted(found_item.rglob("*"))
            if p.is_file()
        }
        dir_classes = sorted(
            c
            for c, home in class_home.items()
            if home.startswith(f"models/{arch}/") and _stem_key(home) not in layer_stems
        )
        regs = {a: c for a, c in registered.items() if c in dir_classes}
        req, sup = MODEL_FEATURES.get(arch, ([], []))
        reg_tested = [c for c in set(regs.values()) if c in tested]
        tests = (
            set().union(*[tested[c] for c in reg_tested])
            if reg_tested and len(reg_tested) == len(set(regs.values()))
            else set()
        )
        put(
            {
                "id": f"model:{arch}",
                "kind": "model",
                "name": f"{arch} (hand-written model)",
                "architecture": arch,
                "features": req,
                "supports": sup,
                "files": files,
                "evidence": {
                    "classes": dir_classes,
                    "registered_architectures": regs,
                    "final_overriders": {
                        c: final_by_class.get(c) or final_overriders(c, bases, defined)
                        for c in set(regs.values())
                    },
                    "untested_registered_classes": sorted(set(regs.values()) - set(reg_tested)),
                    "emit_when_models_missing": True,
                },
                "notes": "" if req else "checklist rules for this family are not written yet",
            },
            tests,
        )

    # -- 6. hooks --------------------------------------------------------------
    # pylint: disable-next=too-many-nested-blocks
    for cls in sorted(defs):
        home = class_home.get(cls, "")
        if _stem_key(home) in layer_stems or not (
            cls in FRAMEWORK_CLASSES or home.startswith("models/")
        ):
            continue
        arch = "base" if cls in FRAMEWORK_CLASSES else home.split("/")[1]
        for meth, items in sorted(defs[cls].items()):
            for idx, (rel, dfn) in enumerate(items):
                qualified = f"{cls}::{meth}"
                hid = f"hook:{qualified}" + (f"#{idx + 1}" if idx else "")
                req, sup = HOOK_FEATURES.get(qualified, ([], []))
                if not req and meth in STRUCTURAL:
                    for prefix, feat in WHOLE_MODEL_PREFIX.items():
                        if cls.startswith(prefix):
                            req = [feat]
                mreq, msup = MODEL_FEATURES.get(arch, ([], []))
                used = layer_types_used(dfn.body)
                put(
                    {
                        "id": hid,
                        "kind": "hook",
                        "name": qualified,
                        "architecture": arch,
                        "hook": meth,
                        "variant": HOOK_VARIANTS.get(qualified, ""),
                        "role": "structural" if req else "model-specific",
                        "features": req,
                        "supports": sup,
                        "code": dfn.text,
                        "requires": sorted(
                            f"layer:{t}" for t in used if any(t in v for v in file_types.values())
                        ),
                        "evidence": {
                            "file": rel,
                            "lines": [dfn.start_line, dfn.end_line],
                            "class": cls,
                            "signature": dfn.signature,
                            "header": class_home.get(cls, ""),
                            "layer_types_used": used,
                            "model_features": sorted(set(mreq + msup)),
                        },
                        "source_ref": f"{rel}:{dfn.start_line}-{dfn.end_line}@{ref}",
                    },
                    exercised.get((cls, meth), set()),
                )

    # -- 7. layers -------------------------------------------------------------
    stems: Dict[str, Dict[str, str]] = defaultdict(dict)
    for rel, text in src.items():
        if _stem_key(rel) in layer_stems and not Path(rel).name.startswith("unittest_"):
            stems[_stem_key(rel)][rel] = text
    layer_tests: Dict[str, Set[str]] = {}
    for stem, files in stems.items():
        types = {t for rel in files for t in file_types.get(rel, [])}
        layer_tests[stem] = (
            set().union(*[type_tests.get(t, set()) for t in types]) if types else set()
        )
    # Header-only support code (no layer type) counts as tested when a tested
    # layer includes it.
    for stem, files in stems.items():
        if any(file_types.get(r) for r in files):
            continue
        names = {Path(r).name for r in files}
        inc = re.compile(
            r'#include\s*[<"](?:[\w/]*/)?(' + "|".join(re.escape(n) for n in names) + r')[>"]'
        )
        for other, ofiles in stems.items():
            if layer_tests.get(other) and any(inc.search(t) for t in ofiles.values()):
                layer_tests[stem] = layer_tests[stem] | layer_tests[other]
    for stem, files in sorted(stems.items()):
        types = sorted({t for rel in files for t in file_types.get(rel, [])})
        origin = "layers" if stem.startswith("layers/") else "models"
        lid = f"layer:{types[0]}" if len(types) == 1 else f"layer:{stem.split('/')[-1]}"
        put(
            {
                "id": lid,
                "kind": "layer",
                "name": stem.split("/")[-1],
                "architecture": stem.split("/")[1] if origin == "models" else "shared",
                "files": files,
                "origin": origin,
                "evidence": {
                    "layer_types": types,
                    "created_by_exercised": (
                        sorted(set().union(*[type_users.get(t, set()) for t in types]))
                        if types
                        else []
                    ),
                    "emit_when_models_missing": origin == "models",
                    "note": (
                        ""
                        if origin == "models"
                        else "shared layer in Applications/CausalLM/layers; "
                        "assumed present, not emitted"
                    ),
                },
            },
            layer_tests[stem],
        )

    # -- 8. converters ---------------------------------------------------------
    # A reference fixture generator writes the nntr_*.bin the reference test
    # loads; when it imports a res/ converter to do so, that converter is
    # exercised by the same test.
    gen_dir = tests_dir / "causallm_reference" / "generators"
    gen_tests: Dict[str, Set[str]] = defaultdict(set)
    for layer_type, fixture in fixtures.items():
        for handle in fixture:
            gen_tests[handle.replace("_tiny", "")].add(layer_type)
    res_tests: Dict[str, Set[str]] = defaultdict(set)
    generators = sorted(gen_dir.glob("generate_*_reference.py")) if gen_dir.is_dir() else []
    for path in generators:
        text = _read(path)
        key = path.stem[len("generate_"):-len("_reference")]
        for match in re.finditer(r'"res"\s*/\s*"([^"]+)"(?:\s*/\s*"([^"]+)")?', text):
            res_dir = "/".join(x for x in match.groups() if x)
            for mod in re.findall(r"from\s+(\w+)\s+import", text):
                res_tests[f"{res_dir}/{mod}.py"] |= gen_tests.get(key, set())
    for path in sorted((cl_path / "res").rglob("*.py")):
        rel = path.relative_to(cl_path / "res").as_posix()
        model = next((m for pre, m in CONVERTER_MODEL.items() if rel.startswith(pre)), "")
        req, sup = _layout(MODEL_FEATURES.get(model, ([], [])))
        tests = res_tests.get(rel, set())
        put(
            {
                "id": f"converter:res/{rel}",
                "kind": "converter",
                "name": rel,
                "architecture": model or rel.split("/")[0],
                "variant": "gguf" if "gguf" in path.name else "",
                "features": req,
                "supports": sup,
                "code": _read(path),
                "origin": "res",
                "source_ref": f"res/{rel}@{ref}",
                "evidence": {
                    "path": f"Applications/CausalLM/res/{rel}",
                    "note": (
                        "Used by a reference fixture generator."
                        if tests
                        else "No reference test loads this converter's output."
                    ),
                },
            },
            tests,
        )
    for path in generators:
        model_key = path.stem[len("generate_"):-len("_reference")]
        model = GENERATOR_MODEL.get(model_key, "")
        req, sup = _layout(MODEL_FEATURES.get(model, ([], [])))
        put(
            {
                "id": f"converter:fixture-generator/{model_key}",
                "kind": "converter",
                "name": f"{path.name} (reference fixture generator)",
                "architecture": model or model_key,
                "variant": "fixture",
                "features": req,
                "supports": sup,
                "code": _read(path),
                "origin": "tests",
                "source_ref": f"{path.relative_to(root).as_posix()}@{ref}",
                "evidence": {
                    "path": path.relative_to(root).as_posix(),
                    "note": (
                        "Writes the nntr_*.bin and HF reference logits the reference test compares."
                    ),
                },
            },
            gen_tests.get(model_key, set()),
        )

    core = ensure_core_layer_types(bank, root)
    summary = {"nntrainer_root": str(root), "commit": ref, "bank": str(bank.path),
               "writes": dict(counts), "not_verified": sorted(not_verified), "stats": bank.stats(),
               "reference_tested_classes": {c: sorted(t) for c, t in sorted(tested.items())},
               "nntrainer_core_layer_types": len(core)}
    return summary


CORE_TYPES_META = "nntrainer_core_layer_types"


def core_layer_types(nntrainer_root: Path) -> List[str]:
    """Layer types nntrainer itself provides (nntrainer/layers/**: the
    `type = "..."` of each layer class): layer_normalization, embedding, ..."""
    types = set()
    layers = Path(nntrainer_root) / "nntrainer" / "layers"
    for path in layers.rglob("*.h") if layers.is_dir() else []:
        types.update(find_layer_types(_read(path)))
    return sorted(types)


def ensure_core_layer_types(bank: CodeBank, nntrainer_root: Optional[Path]) -> List[str]:
    """Stored in the bank so generation works without the checkout; refreshed
    from the checkout when it is available."""
    if nntrainer_root:
        types = core_layer_types(Path(nntrainer_root))
        if types and types != bank.get_meta(CORE_TYPES_META):
            bank.set_meta(CORE_TYPES_META, types)
    return bank.get_meta(CORE_TYPES_META, []) or []


def _stem_key(rel: str) -> str:
    return re.sub(r"\.(h|hpp|cpp)$", "", rel)


def _registrations(main_cpp: Path) -> Dict[str, str]:
    """registerModel("Arch", ... make_unique<causallm::Class>) pairs."""
    if not main_cpp.exists():
        return {}
    text = _read(main_cpp)
    out = {}
    for match in re.finditer(r'registerModel\(\s*"([^"]+)"(.*?)\);', text, re.S):
        cls_match = re.search(r"make_unique<\s*causallm::(\w+)\s*>", match.group(2))
        if cls_match:
            out[match.group(1)] = cls_match.group(1)
    return out


# pylint: disable-next=too-many-locals
def _reachable(cls: str, bases: Dict[str, List[str]], defined: Dict[str, set],
               defs: Dict[str, Dict[str, list]], call_names: Set[str]) -> Set[Tuple[str, str]]:
    """(Class, method) pairs that can run when a `cls` object is built and
    the test calls `call_names` on it."""
    def hierarchy(class_name: str, seen: Optional[Set[str]] = None) -> Set[str]:
        seen = set() if seen is None else seen
        seen.add(class_name)
        for builder in bases.get(class_name, []):
            if builder not in seen:
                hierarchy(builder, seen)
        return seen

    views: Dict[str, Dict[str, str]] = {}

    def finals(view: str) -> Dict[str, str]:
        if view not in views:
            views[view] = final_overriders(view, bases, defined)
        return views[view]

    todo: List[Tuple[str, str, str]] = []
    for k in hierarchy(cls):
        if k in defined.get(k, ()):                     # constructor
            todo.append((k, k, k))
    top = finals(cls)
    for name in call_names:
        if name in top:
            todo.append((top[name], name, cls))
    seen: Set[Tuple[str, str, str]] = set()
    while todo:
        item = todo.pop()
        if item in seen:
            continue
        seen.add(item)
        owner, meth, view = item
        for _rel, data in defs.get(owner, {}).get(meth, []):
            qualified, plain = calls_in(data.body)
            for klass, method in qualified:
                if method in defined.get(klass, ()):
                    todo.append((klass, method, view))
            finals_view = finals(view)
            for method in plain:
                if method in finals_view:
                    todo.append((finals_view[method], method, view))
    return {(o, m) for o, m, _v in seen}


def _calls_made(text: str) -> Set[str]:
    return set(re.findall(r"(?:->|\.)\s*([A-Za-z_]\w*)\s*\(", text))


def _reference_tests(tests_dir: Path, known_classes: Set[str]
                     ) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]], Dict[str, Set[str]]]:
    """Classes each reference test builds, the fixtures it loads, and the
    method names it (or the shared test helpers) calls on models."""
    tested: Dict[str, Set[str]] = defaultdict(set)
    fixtures: Dict[str, List[str]] = {}
    calls: Dict[str, Set[str]] = {}
    if not tests_dir.is_dir():
        return tested, fixtures, calls
    helpers = set()
    for helper in ("causallm_test_utils.cpp", "causallm_test_utils.h"):
        if (tests_dir / helper).exists():
            helpers |= _calls_made(_read(tests_dir / helper))
    for path in sorted(tests_dir.glob("unittest_causallm_*_reference.cpp")):
        text = _read(path)
        for class_name in set(re.findall(r"causallm::(\w+)", text)):
            if class_name in known_classes and class_name not in FRAMEWORK_CLASSES:
                tested[class_name].add(path.name)
        fixtures[path.name] = sorted(set(re.findall(r'"([a-z0-9_]+_tiny)"', text)))
        calls[path.name] = _calls_made(text) | helpers
    return tested, fixtures, calls


def _fixture_atol(fixture_dir: Path, fixtures: Dict[str, List[str]]) -> Optional[float]:
    vals = []
    for names in fixtures.values():
        for name in names:
            meta = fixture_dir / name / "meta.json"
            if meta.exists():
                try:
                    logits_atol_fp32 = json.loads(_read(meta)).get("logits_atol_fp32")
                    if logits_atol_fp32 is not None:
                        vals.append(float(logits_atol_fp32))
                except ValueError:
                    pass
    return max(vals) if vals else None


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m knowledge.capture_causallm",
                                description="Fill the local code bank from Applications/CausalLM.")
    parser.add_argument(
        "--nntrainer-root", help="nntrainer checkout (default: found from this file's location)"
    )
    parser.add_argument(
        "--db", help="database path (default: tools/agentic-ai-workbench/code_bank/code_bank.db)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.nntrainer_root) if args.nntrainer_root else find_nntrainer_root()
    if not root:
        parser.error("could not find the nntrainer checkout; pass --nntrainer-root")
    with CodeBank(args.db) as bank:
        summary = capture(root, bank)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        stats = summary["stats"]
        print(f"Code bank: {summary['bank']}")
        print(f"Captured from {summary['nntrainer_root']} @ {summary['commit']}")
        print(f"Writes: {summary['writes']}")
        for kind, by_kind in sorted(stats["by_kind"].items()):
            print(f"  {kind:<10} " + ", ".join(f"{k}: {v}" for k, v in sorted(by_kind.items())))
        print("Reference-tested classes: " + ", ".join(summary["reference_tested_classes"]))
        print(f"Not verified by a reference test ({len(summary['not_verified'])}): approved only, "
              "never reused until verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
