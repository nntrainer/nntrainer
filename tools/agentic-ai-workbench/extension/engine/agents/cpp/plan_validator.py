"""
Deterministic validation of LLM-generated CausalLM C++.

Runs before the compiler. That ordering matters: the compiler catches syntax
and type errors, but is blind to the failure modes that actually hurt here --
a misspelled layer name compiles cleanly and then silently loads zero weights,
and `swiglu({up, gate})` without the index remap compiles cleanly and silently
computes the wrong function. Those bugs surface as garbage tokens at inference
time, hours later, with nothing pointing back at the cause.

So every rule below checks something the compiler cannot:

  * layer type strings          -> against the real registered catalog
  * property keys               -> against each layer's accepted set
  * layer names                 -> against the weight-loading wire format
  * known-wrong call shapes     -> swiglu remap, mha_core arity
  * invented API symbols        -> the list earlier generators hallucinated
  * unregistered layer usage    -> would throw at runtime, not compile time

Findings are structured, not prose, for two reasons: the correction prompt
feeds on `fix` strings (far more effective on a weak model than a raw compiler
dump), and gaps are written to state.json where the UI and a later auto-fix
pass can consume them without re-parsing C++.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from knowledge.causallm_kb import (
    FORBIDDEN_SYMBOLS,
    LAYER_CATALOG,
    LAYER_IMPL_PROPS,
    NODE_PROPS,
    all_known_types,
    allowed_props,
    unregistered_types,
)

# (layer_type, property) pairs owned by a dedicated rule further down, which
# reports them with a more actionable message than the generic property check.
_DEDICATED_RULE_KEYS = frozenset({
    ("rms_norm", "feature_size"),
})

# Built-in nntrainer types used by hand-written CausalLM models but outside the
# small set the catalog documents for generation. Listed so the checker does
# not flag legitimate code; they are intentionally NOT in the generation menu,
# because the resolver never plans them.
_BUILTIN_EXTRA = frozenset({
    "split", "concat", "reshape", "identity", "permute", "transpose",
    "flatten", "layer_normalization", "embedding", "conv2d", "pooling2d",
    "batch_normalization", "dropout", "loss", "cross_entropy",
})

# createLayer("type", { ... }) -- captures the type string and its property
# block. The body deliberately excludes braces rather than using a lazy
# wildcard: a lazy `.*?` under DOTALL happily runs past the end of one
# createLayer call and swallows the next, which produced properties being
# attributed to the wrong layer. Excluding braces means nested-brace calls
# simply do not match, which costs a little coverage and buys correctness --
# the right trade for a checker, since a false positive on hand-written code
# destroys trust in every other finding.
_CREATE_LAYER = re.compile(
    r'createLayer\(\s*"([A-Za-z0-9_]+)"\s*,\s*\{([^{}]*)\}',
)
# Type-only match, used where the property block is irrelevant.
_CREATE_LAYER_LOOSE = re.compile(r'createLayer\(\s*"([A-Za-z0-9_]+)"')

_WITHKEY = re.compile(r'withKey\(\s*"([A-Za-z0-9_]+)"')
# Raw "key=value" property style, still valid and used by older emitters.
_RAWPROP = re.compile(r'"([A-Za-z0-9_]+)=')
# Kept single-line for the same reason as above.
_NAME_PROP = re.compile(r'withKey\(\s*"name"\s*,\s*([^\n]*?)\)\s*[,}]')
_GAP_SENTINEL = re.compile(r'//\s*@workbench-gap\s*(\{.*?\})\s*$', re.MULTILINE)
_METHOD_DEF = re.compile(
    r'\b(?:Tensor|void|std::pair<Tensor,\s*Tensor>)\s+'
    r'(\w+)::(\w+)\s*\(', re.MULTILINE
)
# registerFactory<causallm::FooLayer> / registerFactory<FooLayer>. A file that
# registers a type is defining it locally (models/<arch>/<arch>_layer.h), so
# that type is legitimate even though it is not in layers/.
_REGISTERED_LOCALLY = re.compile(r'createLayer\s*<\s*(?:\w+::)?(\w+)\s*>')


def _finding(kind: str, detail: str, fix: str = "",
             constraint: str = "", severity: str = "error",
             line: Optional[int] = None) -> Dict:
    f = {"kind": kind, "detail": detail, "severity": severity}
    if fix:
        f["fix"] = fix
    if constraint:
        f["constraint"] = constraint
    if line is not None:
        f["line"] = line
    return f


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------
def _check_layer_types(code: str, strict: bool = True) -> List[Dict]:
    """
    Every createLayer type string must be a real registered type.

    `strict` distinguishes the two callers, which need different answers:

      strict=True  (validating freshly generated code) -- the resolver chose
        every layer from a closed menu, so an off-menu type can only be a
        hallucination. Error.

      strict=False (auditing existing hand-written code) -- model-local layers
        are legitimate and common: several shipping models define a layer next
        to the model file (models/qwen3_moe/qwen3_moe_layer.h) and register it
        from a different translation unit. Warning, not error.

    Note the class name and the type string are NOT mechanically related --
    qwen3_moe registers `causallm::MoELayer` but calls
    `createLayer("qwen_moe")`. So no name-matching heuristic can decide this;
    only the caller knows which mode applies.
    """
    known = all_known_types()
    out: List[Dict] = []
    seen = set()
    for m in _CREATE_LAYER_LOOSE.finditer(code):
        t = m.group(1)
        if t in known or t in seen or t in _BUILTIN_EXTRA:
            continue
        seen.add(t)
        if not strict:
            out.append(_finding(
                "unknown-layer-type",
                f'createLayer("{t}", ...) is not in the catalog. If this is a '
                f'model-local layer, that is fine; if not, it will throw at '
                f'runtime.',
                fix=f'Confirm a layer registers type "{t}".',
                severity="warning",
                line=_line_of(code, m.start()),
            ))
            continue
        # Offer the closest real name -- but only as information. The fix
        # instruction deliberately says "emit a gap marker", not "use this
        # instead", because a plausible-looking substitution is how silently
        # wrong graphs get built.
        near = _closest(t, known)
        hint = f" Closest real type is \"{near}\"." if near else ""
        out.append(_finding(
            "unknown-layer-type",
            f'createLayer("{t}", ...) -- "{t}" is not a registered layer type.'
            f'{hint}',
            fix=f'Do NOT substitute a similar type. Replace this layer with '
                f'the GAP MARKER for op "{t}" so a contributor can implement '
                f'it.',
            constraint="C4",
            line=_line_of(code, m.start()),
        ))
    return out


def _check_props(code: str) -> List[Dict]:
    """Property keys must be accepted by the specific layer they are on."""
    out: List[Dict] = []
    for m in _CREATE_LAYER.finditer(code):
        layer_type, body = m.group(1), m.group(2)
        if layer_type not in all_known_types():
            continue  # already reported by _check_layer_types
        ok = allowed_props(layer_type)
        keys = set(_WITHKEY.findall(body)) | set(_RAWPROP.findall(body))
        for k in sorted(keys - ok):
            # Skip keys a dedicated rule reports with a better message, so the
            # correction prompt gets one clear instruction instead of two
            # overlapping ones -- duplicate findings measurably degrade small
            # models, which try to "fix" the same line twice.
            if (layer_type, k) in _DEDICATED_RULE_KEYS:
                continue
            entry = LAYER_CATALOG.get(layer_type, {})
            if k in LAYER_IMPL_PROPS and not entry.get("layer_impl"):
                reason = (
                    f'"{k}" is a LayerImpl property, but "{layer_type}" is a '
                    f'plain nntrainer::Layer subclass and throws '
                    f'std::invalid_argument on it.'
                )
            else:
                reason = (f'"{k}" is not in the accepted property set for '
                          f'"{layer_type}".')
            out.append(_finding(
                "invalid-property",
                f'createLayer("{layer_type}", ...) is given "{k}". {reason}',
                fix=f'Remove withKey("{k}", ...) from this layer. Accepted: '
                    f'{", ".join(sorted(entry.get("props", {}))) or "(none)"}.',
                constraint="C3",
                line=_line_of(code, m.start()),
            ))
    return out


def _check_names(code: str) -> List[Dict]:
    """
    Layer names are a wire format matched against safetensors keys.

    The single highest-value check in this file: a name containing a '.' means
    the model emitted HuggingFace dotted names, which no weight converter
    produces. It compiles, loads nothing, and yields garbage output.
    """
    out: List[Dict] = []
    for m in _NAME_PROP.finditer(code):
        expr = m.group(1).strip()
        literals = re.findall(r'"([^"]*)"', expr)
        for lit in literals:
            if "." in lit:
                out.append(_finding(
                    "hf-dotted-name",
                    f'Layer name contains a HuggingFace dotted path: "{lit}". '
                    f'Weight loading matches names against converter output, '
                    f'which never emits dotted names. This compiles and then '
                    f'silently loads zero weights.',
                    fix='Use the canonical form, e.g. '
                        '"layer" + std::to_string(layer_id) + "_wq".',
                    constraint="C1",
                    line=_line_of(code, m.start()),
                ))
                break
    return out


def _check_forbidden(code: str) -> List[Dict]:
    """Symbols earlier generators invented. None of them exist."""
    out: List[Dict] = []
    for sym, why in FORBIDDEN_SYMBOLS.items():
        # Match as a whole token where the symbol is identifier-like.
        pattern = (re.escape(sym) if not sym.isidentifier()
                   else rf'\b{re.escape(sym)}\b')
        m = re.search(pattern, code)
        if m:
            out.append(_finding(
                "forbidden-symbol",
                f'Uses "{sym}", which does not exist. {why}',
                fix=why,
                constraint="C4",
                line=_line_of(code, m.start()),
            ))
    return out


def _check_swiglu(code: str) -> List[Dict]:
    """
    swiglu must be called as swiglu({up, gate}, {1, 0}).

    Compiles either way; wrong numerics silently. Worth a dedicated rule.
    """
    if "swiglu" not in code:
        return []
    calls = re.findall(r'\bswiglu\s*\(\s*\{([^}]*)\}\s*(?:,\s*\{([^}]*)\})?',
                       code)
    out: List[Dict] = []
    for inputs, remap in calls:
        normalized = re.sub(r'\s+', '', remap or '')
        if normalized != "1,0":
            out.append(_finding(
                "swiglu-remap",
                f'swiglu is called with inputs {{{inputs.strip()}}} and remap '
                f'{{{remap.strip() or "<missing>"}}}. The {{1, 0}} index remap '
                f'is mandatory: the layer reads input[0] as gate, but '
                f'nntrainer stores MLP weights in up,gate order. Without it '
                f'this compiles and computes the wrong function.',
                fix='Write exactly: Tensor act = swiglu({up, gate}, {1, 0});',
                constraint="C5",
            ))
    return out


def _check_mha(code: str) -> List[Dict]:
    """
    mha_core needs 5 inputs in {q, k, v, cache_k, cache_v} order.

    Keyed off attention call sites rather than the presence of the literal
    "mha_core", so a body that calls an `mha` handle but never created the
    layer -- or created it under the wrong type -- is still checked. Guarding
    on the type string let exactly that case through silently.
    """
    call_sites = list(
        re.finditer(r'\b(\w*mha\w*)\s*\(\s*\{([^}]*)\}\s*\)', code)
    )
    if "mha_core" not in code and not call_sites:
        return []
    out: List[Dict] = []
    for m in call_sites:
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if len(args) != 5:
            out.append(_finding(
                "mha-arity",
                f'mha_core handle called with {len(args)} input(s): '
                f'{{{", ".join(args)}}}. CausalLM uses external-cache mode, '
                f'which requires exactly 5: {{q, k, v, cache_k, cache_v}}. '
                f'With fewer, the layer allocates its own cache and the host '
                f'KV cache is never bound.',
                fix='Call mha({q, k, v, cache_k, cache_v}) with cache_k/cache_v '
                    'from createKVCachePlaceholders(layer_id, n_heads).',
                line=_line_of(code, m.start()),
            ))
        elif not any("cache" in a for a in args[3:]):
            out.append(_finding(
                "mha-cache-order",
                f'mha_core inputs 4 and 5 are {{{args[3]}, {args[4]}}}, which '
                f'do not look like KV cache placeholders. Input order is fixed.',
                fix='Inputs must be {q, k, v, cache_k, cache_v} in that order.',
                line=_line_of(code, m.start()),
            ))
    # Only require the helper when a call site is actually in external-cache
    # mode (5 inputs). mha_core also supports 3-4 inputs, where the layer owns
    # its cache and the helper is correctly absent -- Gemma4 does exactly that,
    # and demanding the helper there is a false positive.
    external_mode = any(
        len([a for a in m.group(2).split(",") if a.strip()]) == 5
        for m in re.finditer(r'\b(\w*mha\w*)\s*\(\s*\{([^}]*)\}\s*\)', code)
    )
    if external_mode and not re.search(r"KVCachePlaceholders\s*\(", code):
        out.append(_finding(
            "missing-kv-helper",
            'Calls mha_core with 5 inputs (external-cache mode) but never '
            'calls createKVCachePlaceholders(). The cache placeholders must '
            'come from that base-class helper -- hand-built Tensors break '
            'tensor-pool in-place behaviour on ARM.',
            fix='auto [cache_k, cache_v] = '
                'createKVCachePlaceholders(layer_id, n_heads);',
        ))
    # The helper takes the FULL head count; it reduces by GQA internally.
    # Second arg is matched without crossing a newline, since a permissive
    # [^)]+ ran on through trailing comments and reported garbage.
    for m in re.finditer(
        r'createKVCachePlaceholders\(\s*[^,\n]+,\s*([^),\n]+)\)', code
    ):
        arg = m.group(1).strip()
        if "GQA" in arg or "/" in arg:
            out.append(_finding(
                "kv-helper-arg",
                f'createKVCachePlaceholders(..., {arg}) passes an '
                f'already-GQA-reduced head count. The helper applies the '
                f'reduction internally, so this halves the cache twice.',
                fix='Pass n_heads directly: '
                    'createKVCachePlaceholders(layer_id, n_heads).',
                line=_line_of(code, m.start()),
            ))
    return out


def _check_unregistered(code: str) -> List[Dict]:
    """
    Layers not registered by the base class must be registered explicitly.

    This is a runtime throw, not a compile error, so nothing else catches it.
    """
    out: List[Dict] = []
    used = {m.group(1) for m in _CREATE_LAYER_LOOSE.finditer(code)}
    for t in sorted(used & set(unregistered_types())):
        cls = LAYER_CATALOG[t]["cls"]
        short = cls.split("::")[-1]
        if f"createLayer<{cls}>" in code or f"createLayer<{short}>" in code:
            continue
        out.append(_finding(
            "missing-registration",
            f'Uses createLayer("{t}", ...) but never registers {cls}. The '
            f'base class does not register this type, so createLayer() throws '
            f'std::invalid_argument at runtime.',
            fix=f'In registerCustomLayers(), add: '
                f'app_context->registerFactory('
                f'nntrainer::createLayer<{cls}>); wrapped in '
                f'try/catch (std::invalid_argument).',
            constraint="C6",
        ))
    return out


def _check_rms_norm_feature_size(code: str) -> List[Dict]:
    """
    Plain rms_norm throws on `feature_size`; only reshaped_rms_norm takes it.

    Easy mistake because the two layers look interchangeable and the reshaped
    variant always carries feature_size.
    """
    out: List[Dict] = []
    for m in _CREATE_LAYER.finditer(code):
        if m.group(1) != "rms_norm":
            continue
        if "feature_size" in m.group(2):
            out.append(_finding(
                "rms-norm-feature-size",
                'createLayer("rms_norm", ...) is given "feature_size". Plain '
                'rms_norm normalises the full width and throws '
                'std::invalid_argument on that property.',
                fix='Either drop feature_size, or switch the type to '
                    '"reshaped_rms_norm" if per-head normalisation was '
                    'intended.',
                constraint="C3",
                line=_line_of(code, m.start()),
            ))
    return out


def _check_scaffolding(code: str, plan: Dict) -> List[Dict]:
    """The model must not emit a standalone program or invent methods."""
    out: List[Dict] = []
    if re.search(r'\bint\s+main\s*\(', code):
        out.append(_finding(
            "standalone-main",
            'Emitted an int main(). CausalLM model files are components '
            'linked into the causallm library; they have no entry point.',
            fix='Delete main(). Emit only the overridden methods.',
        ))
    expected = {h["hook"] for h in plan.get("holes", [])}
    if expected:
        defined = {m.group(2) for m in _METHOD_DEF.finditer(code)}
        for extra in sorted(defined - expected - {plan.get("class_name", "")}):
            if extra.startswith("~") or extra in ("registerCustomLayers",):
                continue
            out.append(_finding(
                "unexpected-method",
                f'Defines {extra}(), which is not one of the planned holes '
                f'({", ".join(sorted(expected))}).',
                fix=f'Remove {extra}(). Override only the planned hooks -- the '
                    f'base class implementation is correct for everything else.',
                severity="warning",
            ))
        for missing in sorted(expected - defined):
            out.append(_finding(
                "unfilled-hole",
                f'Planned hook {missing}() was never defined.',
                fix=f'Add the {missing}() override.',
            ))
    return out


def _closest(needle: str, haystack) -> Optional[str]:
    """Cheap nearest-name suggestion, for diagnostics only."""
    import difflib
    matches = difflib.get_close_matches(needle, sorted(haystack), n=1,
                                        cutoff=0.6)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Gap extraction
# ---------------------------------------------------------------------------
def extract_gaps(code: str) -> List[Dict]:
    """
    Pull the machine-readable `// @workbench-gap {...}` sentinels out.

    Emitting these as structured JSON rather than leaving bare `// TODO`
    comments is what lets the UI list them as contributor tasks and lets a
    later auto-fix pass act on them without re-parsing C++.
    """
    gaps: List[Dict] = []
    for m in _GAP_SENTINEL.finditer(code):
        try:
            entry = json.loads(m.group(1))
            entry["line"] = _line_of(code, m.start())
            gaps.append(entry)
        except json.JSONDecodeError:
            gaps.append({
                "op": "<unparseable>",
                "raw": m.group(1),
                "line": _line_of(code, m.start()),
            })
    return gaps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def validate(code: str, plan: Dict, strict: bool = True) -> Dict:
    """
    Validate generated C++ against the catalog and the plan.

    Returns
        {
          "ok":       bool,     # no error-severity findings
          "findings": [...],    # structured, ready for correction_prompt
          "gaps":     [...],    # marked contributor tasks
          "counts":   {...},
        }
    """
    findings: List[Dict] = []
    findings += _check_layer_types(code, strict=strict)
    findings += _check_props(code)
    findings += _check_names(code)
    findings += _check_forbidden(code)
    findings += _check_swiglu(code)
    findings += _check_mha(code)
    findings += _check_unregistered(code)
    findings += _check_rms_norm_feature_size(code)
    findings += _check_scaffolding(code, plan)

    gaps = extract_gaps(code)
    errors = [f for f in findings if f.get("severity") == "error"]

    return {
        "ok": not errors,
        "findings": findings,
        "gaps": gaps,
        "counts": {
            "errors": len(errors),
            "warnings": len(findings) - len(errors),
            "gaps": len(gaps),
        },
    }


def format_report(result: Dict) -> str:
    """Human-readable summary for the workbench log."""
    c = result["counts"]
    if result["ok"] and not c["gaps"]:
        return "Validation passed: no findings."
    lines = [
        f"Validation: {c['errors']} error(s), {c['warnings']} warning(s), "
        f"{c['gaps']} gap(s)."
    ]
    for f in result["findings"]:
        loc = f" (line {f['line']})" if f.get("line") else ""
        lines.append(f"  [{f['severity']}] {f['kind']}{loc}: {f['detail']}")
        if f.get("fix"):
            lines.append(f"      fix: {f['fix']}")
    for g in result["gaps"]:
        lines.append(f"  [gap] op '{g.get('op')}' at line {g.get('line')} "
                     f"-- needs a layer implementation")
    return "\n".join(lines)
