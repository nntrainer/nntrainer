"""
Compatibility Agent (LLM optional).

Builds the model architecture graph *without downloading any weights*:
uses AutoModel.from_config(config) instead of AutoModel.from_pretrained(),
which constructs the same module tree (same layer types, same shapes)
with randomly-initialized tensors, entirely from the config already
fetched by the Model Discovery Agent. That's the actual fix for "graph
construction feels slow" -- the slowness was never the tree-walk or the
op-table lookup, it was AutoModel.from_pretrained() silently pulling
down the full pretrained weights just to inspect module types. Real
weights are fetched separately and concurrently by the Weight Download
Agent, which the orchestrator runs in a background thread.

Also checks the workspace cache first: if this model's graph was built
within the last 30 days, skip rebuilding it entirely.
"""
from api.compatibility.op_level_checker import OpLevelCompatibilityChecker
from api.compatibility.semantic_capabilities import describe as describe_semantic_capabilities
from api.parsers.generic_fx_parser import (
    GenericFxParser, TracingError, TransformerSemanticLoweringRequired,
)
from api.adapters.registry import KNOWN_MODEL_TYPES, select_adapter
from api.graph.graph import Graph

from . import cache
from .events import bus

# Bump whenever the shape of what gets cached changes in a way that would
# make an old cache entry misleading rather than just "a bit stale" -- e.g.
# adding semantic_ir/semantic_capabilities in this version. An old v1 cache
# entry has no semantic_ir key at all, which is indistinguishable from "we
# checked and there's genuinely no adapter for this architecture" unless
# the version is compared explicitly.
COMPATIBILITY_CACHE_SCHEMA_VERSION = 2


def run(state: dict) -> dict:
    bus.agent_status("compatibility", "running")
    model_name = state["model_name"]
    out_dir = state["out_dir"]
    cache_root = cache.cache_root_for(out_dir, model_name)
    meta = cache.read_meta(cache_root)

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        bus.agent_status("compatibility", "error", "transformers not installed")
        return state

    # Fetching the config is cheap (already-cached HTTP call, no weights) --
    # doing it before the cache-freshness decision, rather than after, is
    # what lets that decision account for whether this is a known
    # transformer architecture at all (see the schema-version check below).
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        bus.log(f"Failed to fetch config for '{model_name}': {exc}", "error")
        bus.agent_status("compatibility", "error", str(exc))
        state.setdefault("errors", []).append(f"compatibility: {exc}")
        return state

    is_known_architecture = getattr(config, "model_type", None) in KNOWN_MODEL_TYPES

    if cache.is_fresh(meta, "graph"):
        cache_entry = meta.get("graph", {})
        stale_schema = cache_entry.get("schema_version") != COMPATIBILITY_CACHE_SCHEMA_VERSION
        missing_required_semantic_ir = is_known_architecture and not cache_entry.get("has_semantic_ir")

        if stale_schema or missing_required_semantic_ir:
            reason = (
                "cache schema is out of date" if stale_schema
                else f"cached entry is missing semantic IR for known architecture "
                     f"'{getattr(config, 'model_type', None)}'"
            )
            bus.log(f"Ignoring cached compatibility result: {reason} -- rebuilding", "info")
        else:
            loaded = _load_cached_graph(cache_root)
            if loaded:
                state["graph_ir"], state["report"], state["semantic_ir"], state["semantic_capabilities"] = loaded
                age = cache.age_days(meta, "graph")
                bus.log(f"Using cached compatibility result ({age:.1f} days old, cache expires after 30 days)")
                summary = state["report"]["summary"]
                _maybe_suggest(state, state["report"]["unsupported"])
                bus.agent_status(
                    "compatibility", "done",
                    f"{summary['compatibility']}% (cached, {age:.1f}d old)",
                )
                return state

    try:
        # Config was already fetched above; building the module tree from
        # it needs no weight download either.
        model = AutoModel.from_config(config)
        model.eval()
        bus.log("Built module tree from config only (AutoModel.from_config) -- no weight download needed for this step")
    except Exception as exc:
        bus.log(f"Failed to build module tree for '{model_name}': {exc}", "error")
        bus.agent_status("compatibility", "error", str(exc))
        state.setdefault("errors", []).append(f"compatibility: {exc}")
        return state

    # Architecture adapter selection happens first and doesn't depend on
    # module-tree tracing at all -- it reads config + real submodules
    # (model.layers[i].self_attn.q_proj, etc.) directly, which is exactly
    # the information hook-based tracing struggles to recover for
    # transformer residuals/concats. See api/adapters and api/semantic.
    adapter = select_adapter(config, model)
    if adapter is not None:
        try:
            semantic_ir = adapter.build_semantic_ir(config, model)
            state["semantic_ir"] = semantic_ir.to_dict()
            bus.log(
                f"Matched architecture adapter '{adapter.__class__.__name__}' -- "
                f"built semantic IR ({semantic_ir.num_layers} decoder layers)"
            )
            state["semantic_capabilities"] = describe_semantic_capabilities(semantic_ir)
        except Exception as exc:
            # STRICT: this is a known, supported architecture (an adapter
            # matched it) -- silently falling back to the module-order
            # graph here would produce exactly the confidently-wrong C++
            # this whole redesign exists to avoid. Stop instead of
            # degrading quietly.
            bus.log(
                f"Semantic conversion failed for supported architecture "
                f"'{getattr(config, 'model_type', None)}' via "
                f"'{adapter.__class__.__name__}': {exc}. C++ generation has "
                f"been stopped -- this architecture is not eligible for the "
                f"inspection-only module-graph fallback.",
                "error",
            )
            bus.agent_status("compatibility", "error", f"semantic adapter failed: {exc}")
            state["semantic_ir"] = None
            state.setdefault("errors", []).append(f"compatibility: semantic adapter failed: {exc}")
            return state
    else:
        state["semantic_ir"] = None
        bus.log(
            f"No semantic adapter matched model_type="
            f"{getattr(config, 'model_type', None)!r} -- falling back to "
            f"inspection-only module-tree graph (not executable C++; see "
            f"api/adapters to add support for this architecture)",
            "info",
        )

    try:
        graph = GenericFxParser(model, model_name=model_name).parse()
        bus.log("Traced model using module-tree walker (no torch.fx dependency)")
    except TransformerSemanticLoweringRequired as exc:
        # Expected for known transformer architectures whenever hook
        # tracing can't fully connect the graph -- the semantic IR built
        # above (if adapter selection succeeded) is the real source of
        # truth for this model, and the module-tree graph is demoted to
        # an optional inspection view rather than aborting the pipeline.
        bus.log(str(exc), "info")
        state["graph_ir"] = None
        state["report"] = {"summary": {"compatibility": None, "supported_nodes": 0, "unsupported_nodes": 0}, "unsupported": []}
        if state.get("semantic_ir") is None:
            bus.agent_status("compatibility", "error", "semantic lowering required but no adapter matched")
            state.setdefault("errors", []).append(f"compatibility: {exc}")
            return state
        _save_cached_graph(cache_root, None, state["report"], state.get("semantic_ir"), state.get("semantic_capabilities"))
        bus.agent_status("compatibility", "done", "semantic IR built; module-tree view unavailable")
        return state
    except TracingError as exc:
        bus.log(str(exc), "error")
        bus.agent_status("compatibility", "error", "tracing failed")
        state.setdefault("errors", []).append(f"compatibility: {exc}")
        return state

    report = OpLevelCompatibilityChecker().analyze(graph)
    graph_ir = graph.export()
    state["graph_ir"] = graph_ir
    state["report"] = report

    _save_cached_graph(cache_root, graph_ir, report, state.get("semantic_ir"), state.get("semantic_capabilities"))

    summary = report["summary"]
    bus.log(
        f"Traced {len(graph)} nodes -- {summary['supported_nodes']} supported / "
        f"{summary['unsupported_nodes']} unsupported ({summary['compatibility']}% compatibility)"
    )

    unsupported = report["unsupported"]
    if unsupported:
        for node in unsupported:
            bus.log(f"  [unsupported] {node['name']} ({node['type']}) -- {node['reason']}", "warn")
        _maybe_suggest(state, unsupported)

    bus.agent_status(
        "compatibility", "done",
        f"{summary['compatibility']}% compatible" if summary["unsupported_nodes"] == 0
        else f"{summary['unsupported_nodes']} unsupported op(s) -- see log",
    )
    return state


def _maybe_suggest(state: dict, unsupported: list):
    if not unsupported:
        return
    api_key = state.get("anthropic_api_key")
    if api_key:
        state["suggestions"] = _get_llm_suggestions(api_key, unsupported)
    else:
        bus.log(
            "No Anthropic API key configured -- skipping LLM suggestions for unsupported ops "
            "(set aiCompilerWorkbench.anthropicApiKey to enable).",
            "info",
        )


def _load_cached_graph(cache_root: str):
    import os
    import json
    report_path = os.path.join(cache_root, "report.json")
    if not os.path.exists(report_path):
        return None
    try:
        graph_ir = None
        graph_path = os.path.join(cache_root, "graph_ir.json")
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_ir = json.load(f)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        semantic_ir = semantic_capabilities = None
        semantic_path = os.path.join(cache_root, "semantic_ir.json")
        if os.path.exists(semantic_path):
            with open(semantic_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            semantic_ir = cached.get("semantic_ir")
            semantic_capabilities = cached.get("semantic_capabilities")
        return graph_ir, report, semantic_ir, semantic_capabilities
    except Exception:
        return None


def _save_cached_graph(cache_root: str, graph_ir: dict, report: dict, semantic_ir: dict = None, semantic_capabilities: dict = None):
    import os
    import json
    os.makedirs(cache_root, exist_ok=True)
    if graph_ir is not None:
        with open(os.path.join(cache_root, "graph_ir.json"), "w", encoding="utf-8") as f:
            json.dump(graph_ir, f, indent=2)
    with open(os.path.join(cache_root, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(cache_root, "semantic_ir.json"), "w", encoding="utf-8") as f:
        json.dump({"semantic_ir": semantic_ir, "semantic_capabilities": semantic_capabilities}, f, indent=2)
    cache.set_entry(cache_root, "graph", {
        "nodes": len(graph_ir.get("nodes", [])) if graph_ir else 0,
        "schema_version": COMPATIBILITY_CACHE_SCHEMA_VERSION,
        "has_semantic_ir": semantic_ir is not None,
    })


def _get_llm_suggestions(api_key: str, unsupported: list) -> dict:
    """One short LangChain call per unsupported op. Best-effort: a
    failed call for one op just logs a warning, it doesn't abort the
    pipeline."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
    except ImportError:
        bus.log("langchain-anthropic not installed -- skipping LLM suggestions", "warn")
        return {}

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=200, api_key=api_key)
    suggestions = {}

    for node in unsupported:
        prompt = (
            f"In an nntrainer C++ model built from a traced op-level graph, the op "
            f"\"{node['name']}\" (type: {node['type']}) has no direct nntrainer layer "
            f"mapping. Reason given: \"{node['reason']}\". In 2-3 sentences, suggest how "
            f"a developer should handle this (hand-write a custom layer, fuse it into a "
            f"neighboring op, or something else). Be concrete and brief."
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = (resp.content or "").strip()
            suggestions[node["name"]] = text
            bus.log(f"  [suggestion] {node['name']}: {text}")
        except Exception as exc:
            bus.log(f"  [suggestion] {node['name']}: request failed -- {exc}", "warn")

    return suggestions
