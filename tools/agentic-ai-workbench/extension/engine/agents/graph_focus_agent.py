"""
Graph Focus Agent (LLM-optional, no LangChain required for the fast path).

Called as:
    python -m agents.graph_focus_agent <out_dir> <graph_key> <query…>

Resolution order:
  1. Typed reset words         → clear focus immediately
  2. Latency/bottleneck rules  → top-N by estimated_ms from profiler data
  3. Keyword match             → substring on node type / label
  4. LLM fallback              → only if api_key configured

All results are emitted via the shared event bus.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from core.events import bus

logger = logging.getLogger(__name__)

RESET_WORDS = frozenset({
    "reset", "clear", "show all", "full graph", "show everything", "all"
})

BOTTLENECK_WORDS = frozenset({
    "bottleneck", "bottlenecks", "slow", "slowest", "hottest",
    "expensive", "heaviest", "heavy", "largest", "biggest",
})

AGENT_ID = "graph_focus"


# ------------------------------------------------------------------ state helpers
def _load_state(out_dir: str) -> dict:
    p = Path(out_dir) / "state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _graph_nodes(state: dict, graph_key: str) -> list:
    key = "cpp_graph_view" if graph_key == "cpp" else "graph_view"
    return (state.get(key) or {}).get("nodes", [])


# ------------------------------------------------------------------ matchers
def _bottleneck_match(state: dict, graph_key: str, query: str) -> Optional[List[str]]:
    """Return top-N node ids by profiler share, or None if not applicable."""
    if graph_key != "ini":
        return None
    if not any(w in query for w in BOTTLENECK_WORDS):
        return None

    layers = (state.get("device_profile") or {}).get("layers") or []
    if not layers:
        return []   # empty list signals "asked for bottleneck but no data"

    ranked = sorted(layers, key=lambda l: l.get("estimated_ms", 0.0), reverse=True)
    top_n  = max(1, min(5, len(ranked) // 4 or 1))
    top_names = {l["name"] for l in ranked[:top_n]}

    nodes = _graph_nodes(state, graph_key)
    return [n["id"] for n in nodes if n.get("label") in top_names]


def _keyword_match(state: dict, graph_key: str, query: str) -> List[str]:
    """Substring match on node type/label."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", query) if len(t) >= 3]
    if not tokens:
        return []
    nodes = _graph_nodes(state, graph_key)
    return [
        n["id"] for n in nodes
        if any(t in f"{n.get('type','')} {n.get('label','')}".lower() for t in tokens)
    ]


def _llm_match(state: dict, graph_key: str, query: str, api_key: str) -> Tuple[Optional[List[str]], str]:
    """LLM fallback — returns (ids_or_None, reason_string)."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None, "langchain-anthropic not installed"

    nodes = _graph_nodes(state, graph_key)
    if not nodes:
        return None, "no graph data"

    summaries = [{"id": n["id"], "type": n.get("type",""), "label": n.get("label","")} for n in nodes]
    prompt = (
        "Given this list of neural network graph nodes (JSON) and a user request, "
        "respond with ONLY a JSON array of matching node ids. "
        "Empty array [] if nothing matches.\n\n"
        f"Nodes: {json.dumps(summaries)}\n\nRequest: {query}"
    )

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512, api_key=api_key)
        resp = llm.invoke([SystemMessage(content="You are a graph node selector."),
                           HumanMessage(content=prompt)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip().lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        ids = json.loads(text)
        valid = {n["id"] for n in nodes}
        return [i for i in ids if i in valid], ""
    except Exception as exc:
        return None, f"LLM matching failed: {exc}"


# ------------------------------------------------------------------ main
def run(out_dir: str, graph_key: str, query: str) -> dict:
    query_stripped = query.strip()
    query_lower    = query_stripped.lower()

    if not query_stripped:
        bus.graph_focus(graph_key, False, [], "empty query")
        return {"ok": False}

    # 1. Reset
    if query_lower in RESET_WORDS:
        bus.graph_focus(graph_key, True, [], "reset to full graph", source="reset")
        return {"ok": True}

    state = _load_state(out_dir)
    if not _graph_nodes(state, graph_key):
        msg = f"No {graph_key} graph yet — run the pipeline first."
        bus.log(msg, "warn")
        bus.graph_focus(graph_key, False, [], msg)
        return {"ok": False}

    # 2. Bottleneck / latency rules
    bn = _bottleneck_match(state, graph_key, query_lower)
    if bn is not None:
        if bn:
            expl = f"Top {len(bn)} node(s) by estimated compute share"
            bus.log(f"Graph focus ({graph_key}): {expl}")
            bus.graph_focus(graph_key, True, bn, expl, source="rule")
            return {"ok": True}
        msg = "Run 'Run On-Device Profiling' first to use bottleneck queries."
        bus.log(msg, "warn")
        bus.graph_focus(graph_key, False, [], msg)
        return {"ok": False}

    # 3. Keyword match
    kw = _keyword_match(state, graph_key, query_lower)
    if kw:
        expl = f"{len(kw)} node(s) matched by keyword"
        bus.log(f"Graph focus ({graph_key}): {expl}")
        bus.graph_focus(graph_key, True, kw, expl, source="rule")
        return {"ok": True}

    # 4. LLM fallback. Key comes from the environment (the extension passes it
    # that way; it's no longer persisted in state.json). Fall back to state for
    # any older state.json that still has it.
    api_key = os.environ.get("ANTHROPIC_API_KEY") or state.get("anthropic_api_key")
    if not api_key:
        msg = (
            f"No keyword match for '{query_stripped}'. "
            "Set aiCompilerWorkbench.anthropicApiKey to enable LLM-based matching."
        )
        bus.log(msg, "warn")
        bus.graph_focus(graph_key, False, [], msg)
        return {"ok": False}

    bus.log(f"Keyword match empty for '{query_stripped}' — trying LLM …")
    ids, reason = _llm_match(state, graph_key, query_stripped, api_key)
    if ids is None:
        bus.log(reason, "warn")
        bus.graph_focus(graph_key, False, [], reason, source="llm")
        return {"ok": False}
    if not ids:
        msg = f"Nothing matches '{query_stripped}' — try different wording."
        bus.log(msg)
        bus.graph_focus(graph_key, False, [], msg, source="llm")
        return {"ok": False}

    expl = f"{len(ids)} node(s) matched by LLM"
    bus.log(f"Graph focus ({graph_key}): {expl}")
    bus.graph_focus(graph_key, True, ids, expl, source="llm")
    return {"ok": True}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python -m agents.graph_focus_agent <out_dir> <graph_key> <query…>")
        sys.exit(1)
    result = run(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
    print(json.dumps(result))
