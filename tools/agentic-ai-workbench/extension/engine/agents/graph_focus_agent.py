# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file graph_focus_agent.py
# @brief Graph Focus Agent (LLM-optional, no LangChain required for the fast path).
# @author prachi-t17 <prachi.t@samsung.com>

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
from agents.optional_deps import API_MODEL

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
    path = Path(out_dir) / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


#: Webview pane key -> the state keys holding the graph that pane draws,
#: newest first. Node ids in these views are the ids the pane renders, so
#: whatever this agent returns can be highlighted directly. (The older
#: ini/cpp keys only exist in state.json files from earlier runs.)
_PANE_VIEW_KEYS = {
    "ini": ("model_graph_view", "ini_graph_view"),
    "cpp": ("nntrainer_graph_view", "cpp_graph_view"),
}


def _graph_nodes(state: dict, graph_key: str) -> list:
    for key in _PANE_VIEW_KEYS.get(graph_key, ()):
        nodes = (state.get(key) or {}).get("nodes")
        if nodes:
            return nodes
    return []


def _search_text(node: dict) -> str:
    """What a keyword query is matched against: the box's type and label,
    plus its role and HF module path, so e.g. "self attention" also finds
    the Q/K/V projections in the Model Graph."""
    attrs = node.get("attributes") or {}
    return " ".join(str(part) for part in (
        node.get("type", ""), node.get("label", ""),
        node.get("semanticType", ""), attrs.get("hf_module", ""),
    )).lower()


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
    """Substring match on node type/label/role/HF module."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", query) if len(t) >= 3]
    if not tokens:
        return []
    nodes = _graph_nodes(state, graph_key)
    return [n["id"] for n in nodes if any(t in _search_text(n) for t in tokens)]


# pylint: disable-next=too-many-locals
def _llm_match(
    state: dict, graph_key: str, query: str, api_key: str
) -> Tuple[Optional[List[str]], str]:
    """LLM fallback — returns (ids_or_None, reason_string)."""
    try:
        # pylint: disable-next=import-outside-toplevel  # optional package
        from langchain_anthropic import ChatAnthropic
        # pylint: disable-next=import-outside-toplevel  # optional package
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None, "langchain-anthropic not installed"

    nodes = _graph_nodes(state, graph_key)
    if not nodes:
        return None, "no graph data"

    summaries = [
        {"id": n["id"], "type": n.get("type", ""), "label": n.get("label", "")} for n in nodes
    ]
    prompt = (
        "Given this list of neural network graph nodes (JSON) and a user request, "
        "respond with ONLY a JSON array of matching node ids. "
        "Empty array [] if nothing matches.\n\n"
        f"Nodes: {json.dumps(summaries)}\n\nRequest: {query}"
    )

    try:
        llm = ChatAnthropic(model=API_MODEL, max_tokens=512, api_key=api_key)
        resp = llm.invoke([SystemMessage(content="You are a graph node selector."),
                           HumanMessage(content=prompt)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip().lstrip("`").lstrip("json").strip()
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        ids = json.loads(text)
        valid = {n["id"] for n in nodes}
        return [i for i in ids if i in valid], ""
    # model errors vary; Focus falls back to the rules
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"LLM matching failed: {exc}"


# ------------------------------------------------------------------ main
# pylint: disable-next=too-many-return-statements
def run(out_dir: str, graph_key: str, query: str) -> dict:
    """Find the nodes of a saved graph view that match a Focus query."""
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
    bottleneck_match = _bottleneck_match(state, graph_key, query_lower)
    if bottleneck_match is not None:
        if bottleneck_match:
            expl = f"Top {len(bottleneck_match)} node(s) by estimated compute share"
            bus.log(f"Graph focus ({graph_key}): {expl}")
            bus.graph_focus(graph_key, True, bottleneck_match, expl, source="rule")
            return {"ok": True}
        msg = "Run 'Run On-Device Profiling' first to use bottleneck queries."
        bus.log(msg, "warn")
        bus.graph_focus(graph_key, False, [], msg)
        return {"ok": False}

    # 3. Keyword match
    keyword_match = _keyword_match(state, graph_key, query_lower)
    if keyword_match:
        expl = f"{len(keyword_match)} node(s) matched by keyword"
        bus.log(f"Graph focus ({graph_key}): {expl}")
        bus.graph_focus(graph_key, True, keyword_match, expl, source="rule")
        return {"ok": True}

    # 4. LLM fallback. The key comes from the environment (the extension passes
    # it that way and never stores it in state.json); a state.json written by
    # an older version may still hold one.
    api_key = os.environ.get("ANTHROPIC_API_KEY") or state.get("anthropic_api_key")
    if not api_key:
        msg = (
            f"No keyword match for '{query_stripped}'. "
            "Store an Anthropic API key (AI Compiler Workbench: Set Anthropic API Key) to enable "
            "LLM-based matching."
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
