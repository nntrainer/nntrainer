# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file codegen_validator.py
# @brief Checks Claude's generated CausalLM files against the generation plan.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Checks Claude's generated CausalLM files against the generation plan before
anyone sees them (and before they are saved to the code bank).

Errors (the files are sent back to Claude with these):
  * the two classes and their bases (`virtual public Transformer`,
    `public CausalLM, public <Transformer class>`) and the CausalLM class's
    constructor initialising Transformer(..., ModelType::CAUSALLM) first
  * every hook marked INHERIT is not overridden; every hook marked COPY is
    overridden with exactly the bank's verified body (comments, whitespace
    and class names aside); every hook marked WRITE is overridden
  * no setupParameters override (it never runs during construction)
  * registerCustomLayers / constructModel overridden in both classes or neither
  * createLayer types exist, their properties are allowed, each call is named
  * custom layer types are registered (unless the base already registers them)
  * forbidden symbols
  * a written weight converter is valid Python and mentions every tensor
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .causallm_kb import BUILTIN_TYPES, FORBIDDEN_SYMBOLS, LAYER_CATALOG, allowed_props
from .cpp_extract import (
    CppDefinition,
    find_classes,
    find_definitions,
    find_inline_methods,
    find_layer_types,
    mask_non_code,
)

#: Hooks that, when overridden in the Transformer class, need an override in
#: the CausalLM class too (CausalLM overrides them as well, so the final
#: overrider would otherwise be ambiguous or bypassed).
PAIRED_HOOKS = ("registerCustomLayers", "constructModel")


@dataclass
class ValidationResult:
    """The errors and warnings found in Claude's files."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    definitions: Dict[str, Dict[str, CppDefinition]] = field(
        default_factory=dict
    )  # class -> method -> def
    new_layer_types: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def to_dict(self) -> dict:
        """The result as JSON-ready data."""
        return {"ok": self.passed, "errors": self.errors, "warnings": self.warnings,
                "new_layer_types": self.new_layer_types}


def normalise_body(body: str, class_names: Iterable[str] = ()) -> str:
    """Body text with comments, whitespace and the given class names removed,
    so a verbatim copy compares equal whatever its formatting."""
    text = mask_non_code(body)
    # mask_non_code blanks string *contents*; compare those too.
    text = _restore_strings(body, text)
    for class_name in sorted(set(class_names), key=len, reverse=True):
        if class_name:
            text = re.sub(rf"\b{re.escape(class_name)}\b", "@CLASS@", text)
    return re.sub(r"\s+", "", text)


def _restore_strings(original: str, masked: str) -> str:
    out = list(masked)
    for match in re.finditer(r'"(?:[^"\\\n]|\\.)*"', original):
        if masked[match.start()] == '"':
            out[match.start():match.end()] = list(original[match.start():match.end()])
    return "".join(out)


def _calls(code: str, name: str) -> List[Tuple[int, str]]:
    """(start, text) of every `name(...)` call in code (comments ignored)."""
    masked = mask_non_code(code)
    found = []
    for match in re.finditer(rf"\b{name}\s*\(", masked):
        depth, k = 0, match.end() - 1
        while k < len(masked):
            if masked[k] == "(":
                depth += 1
            elif masked[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        found.append((match.start(), code[match.start():k + 1]))
    return found


def _base_registered_types(bank) -> Set[str]:
    """Layer types the base classes register themselves."""
    classes = set()
    for hid in ("hook:Transformer::registerCustomLayers", "hook:CausalLM::registerCustomLayers"):
        entry = bank.get(hid) if bank else None
        if entry:
            classes |= set(re.findall(r"createLayer<\s*causallm::(\w+)\s*>", entry["code"]))
    return {
        t
        for t, info in LAYER_CATALOG.items()
        if str(info.get("cls", "")).rsplit("::", maxsplit=1)[-1] in classes
    }


def _layer_class(type_name: str, spec=None) -> str:
    cls = str(LAYER_CATALOG.get(type_name, {}).get("cls", "")).rsplit("::", maxsplit=1)[-1]
    return cls or (getattr(spec, "layer_classes", {}) or {}).get(type_name, "")


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def validate(files: Dict[str, str], spec, bank=None) -> ValidationResult:
    """`spec` is a codegen.GenSpec."""
    result = ValidationResult()
    transformer_cls, causallm_cls = spec.transformer_class, spec.causallm_class
    header, source = files.get(spec.header_name), files.get(spec.source_name)
    if header is None:
        result.errors.append(f"missing {spec.header_name}")
    if source is None:
        result.errors.append(f"missing {spec.source_name}")
    if spec.converter_action == "write" and "weight_converter.py" not in files:
        result.errors.append(
            "missing weight_converter.py (the plan has no verified converter to reuse)"
        )
    if header is None or source is None:
        return result

    cpp_files = {n: t for n, t in files.items() if n.endswith((".h", ".hpp", ".cpp"))}
    extra = {n: t for n, t in cpp_files.items() if n not in (spec.header_name, spec.source_name)}

    # -- classes ------------------------------------------------------------
    classes = {c.name: c.bases for c in find_classes(header)}
    if transformer_cls not in classes:
        result.errors.append(
            f"{spec.header_name} must declare class {transformer_cls} : virtual public Transformer"
        )
    elif "Transformer" not in classes[transformer_cls] or not re.search(
        rf"class\s+(?:[A-Z_][A-Z0-9_]*\s+)*{transformer_cls}\s*:"
        r"[^{]*\bvirtual\s+public\s+Transformer\b",
        mask_non_code(header),
    ):
        result.errors.append(
            f"{transformer_cls} must derive from Transformer as `virtual public Transformer`"
        )
    if causallm_cls not in classes:
        result.errors.append(
            f"{spec.header_name} must declare class {causallm_cls} : public CausalLM, public"
            f" {transformer_cls}"
        )
    elif not ({"CausalLM", transformer_cls} <= set(classes[causallm_cls])):
        result.errors.append(
            f"{causallm_cls} must derive from both CausalLM and {transformer_cls} (it has"
            f" {classes[causallm_cls]})"
        )

    # -- definitions --------------------------------------------------------
    defs: Dict[str, Dict[str, CppDefinition]] = {}
    for name, text in cpp_files.items():
        found = (find_definitions(text) if name.endswith(".cpp") else []) + find_inline_methods(
            text
        )
        for found_item in found:
            defs.setdefault(found_item.class_name, {})[found_item.method] = found_item
    result.definitions = defs
    gen = {transformer_cls: defs.get(transformer_cls, {}), causallm_cls: defs.get(causallm_cls, {})}

    ctor = gen[causallm_cls].get(causallm_cls)
    if not ctor:
        result.errors.append(
            f"{causallm_cls} needs a constructor {causallm_cls}(json &cfg, json &generation_cfg,"
            " json &nntr_cfg)"
        )
    else:
        sig = re.sub(r"\s+", " ", mask_non_code(ctor.signature))
        tpos, cpos = sig.find(" Transformer("), sig.find(" CausalLM(")
        if tpos == -1 or "ModelType::CAUSALLM" not in sig:
            result.errors.append(
                f"{causallm_cls}'s constructor must initialise the virtual base explicitly: "
                "Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM)"
            )
        elif cpos != -1 and cpos < tpos:
            result.errors.append(
                f"{causallm_cls}'s constructor must list Transformer(...) before CausalLM(...)"
            )
        if f"{transformer_cls}(" not in sig:
            result.errors.append(
                f"{causallm_cls}'s constructor must also initialise {transformer_cls}(cfg,"
                " generation_cfg, nntr_cfg)"
            )

    def defined(hook: str) -> Optional[CppDefinition]:
        return gen[transformer_cls].get(hook) or gen[causallm_cls].get(hook)

    for hook, hook_plan in spec.hooks.items():
        found_item = defined(hook)
        if hook_plan.action == "inherit" and found_item:
            result.errors.append(
                f"do not override {hook}: the plan inherits the verified base implementation "
                f"({hook_plan.entry_id})"
            )
        elif hook_plan.action == "copy":
            if not found_item:
                result.errors.append(
                    f"{hook} must be overridden with the verified code from {hook_plan.entry_id}"
                )
            else:
                want = normalise_body(
                    _body_of(hook_plan.code), [hook_plan.owner_class, transformer_cls, causallm_cls]
                )
                got = normalise_body(
                    found_item.body, [hook_plan.owner_class, transformer_cls, causallm_cls]
                )
                if want != got:
                    result.errors.append(
                        f"{hook} must be copied exactly from {hook_plan.entry_id} (only the class"
                        " qualifier may change); the generated body differs"
                    )
        elif hook_plan.action == "write" and not found_item:
            result.errors.append(f"{hook} must be overridden (the plan marks it as new code)")

    for cls in (transformer_cls, causallm_cls):
        if "setupParameters" in gen[cls]:
            result.errors.append(
                f"{cls} must not override setupParameters (it is never called during construction);"
                f" read extra config keys in {transformer_cls}'s constructor body instead"
            )
    for hook in PAIRED_HOOKS:
        if hook in gen[transformer_cls] and hook not in gen[causallm_cls]:
            result.errors.append(
                f"{transformer_cls} overrides {hook}, so {causallm_cls} must override it too"
            )
    reg = gen[causallm_cls].get("registerCustomLayers")
    if reg and "registerCustomLayers" in gen[transformer_cls]:
        body = mask_non_code(reg.body)
        if (
            "CausalLM::registerCustomLayers" not in body
            or f"{transformer_cls}::registerCustomLayers" not in body
        ):
            result.errors.append(
                f"{causallm_cls}::registerCustomLayers must call CausalLM::registerCustomLayers()"
                f" and {transformer_cls}::registerCustomLayers()"
            )

    # -- layers ---------------------------------------------------------------
    new_types = sorted({t for text in extra.values() for t in find_layer_types(text)})
    result.new_layer_types = new_types
    core = set(getattr(spec, "core_layer_types", []) or [])
    known = (
        set(BUILTIN_TYPES)
        | set(LAYER_CATALOG)
        | set(spec.known_layer_types)
        | core
        | set(new_types)
    )
    used: Set[str] = set()
    for name, text in cpp_files.items():
        if name in extra:
            continue
        for _start, call in _calls(text, "createLayer"):
            match = re.match(r'createLayer\s*\(\s*"([A-Za-z0-9_]+)"', call)
            if not match:
                continue  # e.g. a variable type: checked at build time
            type_name = match.group(1)
            used.add(type_name)
            if type_name not in known:
                result.errors.append(
                    f'unknown layer type "{type_name}" (not an nntrainer or CausalLM layer)'
                )
                continue
            props = re.findall(r'withKey\(\s*"([A-Za-z0-9_]+)"', call)
            if "name" not in props:
                result.errors.append(f'createLayer("{type_name}", ...) has no withKey("name", ...)')
            allowed = (
                allowed_props(type_name)
                if type_name in BUILTIN_TYPES or type_name in LAYER_CATALOG
                else frozenset()
            )
            bad = [p for p in props if allowed and p not in allowed]
            if bad:
                result.errors.append(
                    f'createLayer("{type_name}") does not take {", ".join(sorted(set(bad)))}'
                )
    base_reg = _base_registered_types(bank)
    reg_text = " ".join(
        mask_non_code(d.body)
        for cls in (transformer_cls, causallm_cls)
        for m, d in gen[cls].items()
        if m == "registerCustomLayers"
    )
    for type_name in sorted(used - set(BUILTIN_TYPES) - base_reg - (core - set(LAYER_CATALOG))):
        cls = _layer_class(type_name, spec)
        if type_name in new_types:
            match = re.search(
                r"class\s+(\w+)[^{;]*\{", next(x for x in extra.values() if f'"{type_name}"' in x)
            )
            cls = match.group(1) if match else cls
        if cls and cls not in reg_text:
            result.errors.append(
                f'layer type "{type_name}" is used but {cls} is not registered in'
                ' registerCustomLayers'
            )
    for type_name in new_types:
        result.warnings.append(f'new custom layer "{type_name}" -- review it carefully')

    # -- forbidden symbols ---------------------------------------------------
    for name, text in cpp_files.items():
        masked = mask_non_code(text)
        for sym, why in FORBIDDEN_SYMBOLS.items():
            if sym.endswith(".h"):
                hit = re.search(rf"#include\s*[<\"]{re.escape(sym)}[>\"]", text)
            elif re.fullmatch(r"\w+", sym):
                hit = re.search(rf"(?<![\w:.>]){re.escape(sym)}\s*\(", masked)
            else:
                hit = sym in masked
            if hit:
                result.errors.append(f"{name}: `{sym}` -- {why}")
    if (
        f'#include "{spec.header_name}"' not in source
        and f"#include <{spec.header_name}>" not in source
    ):
        result.errors.append(f"{spec.source_name} must include {spec.header_name}")
    if "causal_lm.h" not in header:
        result.errors.append(f"{spec.header_name} must #include <causal_lm.h>")

    # -- converter -----------------------------------------------------------
    conv = files.get("weight_converter.py")
    if conv is not None:
        try:
            tree = ast.parse(conv)
        except SyntaxError as exc:
            tree = None
            result.errors.append(f"weight_converter.py is not valid Python: {exc}")
        if (
            tree is not None
            and spec.converter_action == "write"
            and not any(
                isinstance(n, ast.FunctionDef)
                and n.name.startswith("save_")
                and n.name.endswith("_for_nntrainer")
                for n in tree.body
            )
        ):
            result.errors.append(
                "weight_converter.py must define save_<model>_for_nntrainer(params, n_layers,"
                " dtype, file) at module level (the verifier calls it)"
            )
        missing = []
        for type_name in spec.tensors:
            leaf = _module_name(type_name)
            if leaf and leaf not in conv:
                missing.append(type_name)
        if missing and not spec.tensors_whole_model:
            result.errors.append("weight_converter.py never handles: " + ", ".join(missing[:12]))
    return result


def _module_name(normalised: str) -> str:
    """layers.*.self_attn.q_norm.weight -> q_norm ; embed_tokens.weight -> embed_tokens"""
    parts = [p for p in normalised.split(".") if p not in ("*", "weight", "bias")]
    return parts[-1] if parts else ""


def _body_of(code: str) -> str:
    """The `{ ... }` body of a captured definition's text."""
    defs = find_definitions(code) or find_inline_methods(code)
    if defs:
        return defs[0].body
    i = code.find("{")
    return code[i:] if i != -1 else code
