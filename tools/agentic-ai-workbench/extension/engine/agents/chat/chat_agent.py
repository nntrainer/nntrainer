"""Chat agent for the webview Chat panel."""
import json
import os
import re
import sys
import subprocess

from ..events import bus


def _build_context_summary(state: dict) -> str:
    """Flatten the pipeline state into a compact text block a single-shot
    LLM call can reason over -- used by the Claude CLI path, which has no
    tool-calling loop of its own."""
    parts = []

    model_name = state.get("model_name")
    architecture = state.get("architecture")
    if model_name or architecture:
        parts.append(f"Model: {model_name}  Architecture: {architecture}")

    summary = (state.get("report") or {}).get("summary")
    if summary:
        parts.append(f"Compatibility summary: {json.dumps(summary)}")

    unsupported = (state.get("report") or {}).get("unsupported", [])
    if unsupported:
        suggestions = state.get("suggestions") or {}
        preview = [
            {"name": n.get("name"), "reason": n.get("reason"),
             "suggestion": suggestions.get(n.get("name"), "")}
            for n in unsupported[:20]
        ]
        parts.append(f"Unsupported ops ({len(unsupported)} total, showing up to 20): "
                      f"{json.dumps(preview)}")

    artifacts = state.get("artifacts") or []
    if artifacts:
        parts.append(f"Artifacts produced: {json.dumps(artifacts)}")

    profile = state.get("profile")
    if profile:
        parts.append(f"Profiling results: {json.dumps(profile)}")

    graph_view = state.get("nntrainer_graph_view") or {}
    nodes = graph_view.get("nodes", [])
    if nodes:
        type_counts: dict = {}
        groups: dict = {}
        for n in nodes:
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            g = n.get("group")
            if g:
                groups.setdefault(g, 0)
                groups[g] += 1
        parts.append(f"Graph node type counts: {json.dumps(type_counts)}")
        parts.append(f"Graph groups (decoder layers etc.) and component counts: "
                      f"{json.dumps(groups)}")

    return "\n\n".join(parts) or "(no state details available)"


def _maybe_layer_detail(state: dict, message: str) -> str:
    """If the question names a specific layer (e.g. 'layer 3'), append that
    layer's full node breakdown so the CLI path can answer without a real
    tool-calling loop."""
    match = re.search(r'\blayer\s*[_#]?\s*(\d+)\b', message, re.IGNORECASE)
    if not match:
        return ""

    graph_view = state.get("nntrainer_graph_view") or {}
    nodes = graph_view.get("nodes", [])
    group = f"decoder_{match.group(1)}"
    layer_nodes = [
        {"label": n.get("label"), "type": n.get("type"), "weightInfo": n.get("weightInfo")}
        for n in nodes if n.get("group") == group
    ]
    if not layer_nodes:
        return ""
    return f"\n\nDetailed structure of {group}: {json.dumps(layer_nodes)}"


def _run_via_claude_cli(state: dict, message: str):
    """
    Zero-config fallback: answer using the local Claude Code CLI instead of
    an Anthropic API key or Cline UMS token. Returns None if the CLI itself
    isn't installed/authenticated (caller then shows the "need a key"
    message); returns an error string if the CLI is present but the call
    failed, so the user still gets a reply instead of silence.
    """
    try:
        from ..cpp.claude_cli_backend import _make_claude_cli_llm
    except ImportError:
        return None

    llm = _make_claude_cli_llm({})
    if llm is None:
        return None

    context = _build_context_summary(state) + _maybe_layer_detail(state, message)
    system = (
        "You are the orchestrator agent for an nntrainer C++ code-generation pipeline "
        "running inside a VS Code extension. Answer the user's question about the most "
        "recent conversion run using ONLY the context provided below. Be concise and "
        "concrete. If the context doesn't contain the answer, say what's missing "
        "instead of guessing."
    )
    user = f"Context:\n{context}\n\nQuestion: {message}"

    try:
        return llm(system, user)
    except Exception as exc:
        return f"Claude CLI request failed: {exc}"


def _ensure_langchain_installed(ums_token: str) -> bool:
    if ums_token:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            return True
        except ImportError:
            pass
    else:
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            return True
        except ImportError:
            pass
    
    bus.chat("assistant", "Installing langchain packages...")
    
    try:
        python_path = sys.executable or "python3"
        pip_args = ["-m", "pip", "install", "-q", "langchain", "langchain-anthropic", "langchain-openai", "langchain-core"]
        subprocess.check_call([python_path] + pip_args)
        bus.chat("assistant", "Langchain packages installed.")
        return True
    except subprocess.CalledProcessError as e:
        bus.chat("assistant", f"Failed to install langchain: {e}")
        return False
    except Exception as e:
        bus.chat("assistant", f"Failed to install langchain: {e}")
        return False


def _load_state(out_dir: str) -> dict:
    state_path = os.path.join(out_dir, "state.json")
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_tools(state: dict):
    from langchain_core.tools import tool

    @tool
    def get_model_summary() -> str:
        """Return the discovered architecture and compatibility summary for the last converted model."""
        return json.dumps({
            "model_name": state.get("model_name"),
            "architecture": state.get("architecture"),
            "compatibility": (state.get("report") or {}).get("summary", {}),
        })

    @tool
    def list_unsupported_ops() -> str:
        """List the ops that don't map to an nntrainer layer yet, with the reason and any LLM suggestion."""
        unsupported = (state.get("report") or {}).get("unsupported", [])
        suggestions = state.get("suggestions") or {}
        for node in unsupported:
            node["suggestion"] = suggestions.get(node["name"], "")
        return json.dumps(unsupported)

    @tool
    def get_artifacts() -> str:
        """List every file the pipeline produced, with size and path."""
        return json.dumps(state.get("artifacts", []))

    @tool
    def get_profile() -> str:
        """Return the profiling results (latency, whether it actually ran)."""
        return json.dumps(state.get("profile", {}))

    @tool
    def get_graph_nodes_by_type(node_type: str) -> str:
        """Return all graph nodes matching a specific type (e.g., 'attention', 'fully_connected', 'normalization', 'residual', 'embedding', 'lm_head').
        Use this to show specific parts of the model like attention layers, MLP layers, or normalization layers."""
        graph_view = state.get("nntrainer_graph_view", {})
        nodes = graph_view.get("nodes", [])
        matching = []
        for n in nodes:
            ntype = n.get("type", "").lower()
            label = n.get("label", "").lower()
            if node_type.lower() in ntype or node_type.lower() in label:
                matching.append({
                    "id": n.get("id"),
                    "label": n.get("label"),
                    "type": n.get("type"),
                    "group": n.get("group"),
                    "weightInfo": n.get("weightInfo"),
                })
        return json.dumps({
            "query": node_type,
            "count": len(matching),
            "nodes": matching,
        })

    @tool
    def get_decoder_layer_structure(layer_index: int) -> str:
        """Return the detailed structure of a specific decoder layer, including all sub-components (attention, norms, MLP).
        Use this to show what's inside a particular layer."""
        graph_view = state.get("nntrainer_graph_view", {})
        nodes = graph_view.get("nodes", [])
        layer_nodes = []
        for n in nodes:
            group = n.get("group", "")
            if group == f"decoder_{layer_index}":
                layer_nodes.append({
                    "id": n.get("id"),
                    "label": n.get("label"),
                    "type": n.get("type"),
                    "template_id": n.get("template_id"),
                    "weightInfo": n.get("weightInfo"),
                })
        if not layer_nodes:
            return json.dumps({"error": f"Layer {layer_index} not found"})
        return json.dumps({
            "layer_index": layer_index,
            "component_count": len(layer_nodes),
            "nodes": layer_nodes,
        })

    return [get_model_summary, list_unsupported_ops, get_artifacts, get_profile, get_graph_nodes_by_type, get_decoder_layer_structure]


def run(out_dir: str, message: str, api_key: str = None, ums_token: str = None):
    state = _load_state(out_dir)

    if not state:
        bus.chat("assistant", "No pipeline has been run yet in this workspace -- click **Run Pipeline** first.")
        return

    if not api_key and not ums_token:
        # Zero-config path: reuse the local Claude Code CLI (same backend
        # agents/cpp/claude_cli_backend.py uses for C++ generation) instead
        # of requiring an Anthropic API key or Cline UMS token.
        cli_reply = _run_via_claude_cli(state, message)
        if cli_reply is not None:
            bus.chat("assistant", cli_reply)
        else:
            bus.chat("assistant",
                     "Chat requires an API key, a Cline UMS token, or the Claude CLI. Set "
                     "aiCompilerWorkbench.anthropicApiKey / clineUmsToken in Settings, or "
                     "install and log in to the Claude CLI "
                     "(npm install -g @anthropic-ai/claude-code && claude login).")
        return

    reply, error = _run_via_langchain(state, message, api_key, ums_token)
    if reply is not None:
        bus.chat("assistant", reply)
        return

    # langchain's agent API has changed across major versions before (the
    # old AgentExecutor/create_tool_calling_agent combo was removed in
    # langchain 1.x in favour of create_agent) and `pip install langchain`
    # always grabs latest, so a version drift like that silently breaks
    # this path again with no way to "reinstall" out of it. Fall back to
    # the Claude CLI -- same backend agents/cpp/claude_cli_backend.py uses
    # -- so chat still answers instead of dead-ending.
    bus.log(f"langchain chat path failed, falling back to Claude CLI: {error}", "warn")
    cli_reply = _run_via_claude_cli(state, message)
    if cli_reply is not None:
        bus.chat("assistant", cli_reply)
    else:
        bus.chat("assistant", f"Chat request failed: {error}")


def _run_via_langchain(state: dict, message: str, api_key: str = None, ums_token: str = None):
    """
    Try the full tool-calling agent via langchain's current create_agent()
    API. Returns (reply, None) on success or (None, error_string) on any
    failure -- install failure, import failure, or runtime exception --
    so the caller can fall back to the Claude CLI instead of dead-ending.
    """
    if not _ensure_langchain_installed(ums_token):
        return None, "langchain packages unavailable"

    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        return None, f"failed to import langchain.agents.create_agent: {exc}"

    tools = _make_tools(state)

    try:
        if ums_token:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="claude-sonnet-4-5-20250929",
                max_tokens=600,
                openai_api_key=ums_token,
                openai_api_base="http://localhost:6543/v1",
            )
        else:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=600, api_key=api_key)
    except ImportError as exc:
        return None, f"failed to import chat model backend: {exc}"

    system_prompt = (
        "You are the orchestrator agent for an nntrainer C++ code-generation pipeline "
        "running inside a VS Code extension. Use the tools to answer questions about "
        "the most recent conversion run. Be concise and concrete."
    )

    try:
        agent = create_agent(llm, tools, system_prompt=system_prompt)
        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
        messages = result.get("messages", [])
        if not messages:
            return None, "agent returned no messages"
        return messages[-1].content, None
    except Exception as exc:
        return None, f"agent invocation failed: {exc}"


if __name__ == "__main__":
    out_dir, message = sys.argv[1], sys.argv[2]
    api_key = os.environ.get("ANTHROPIC_API_KEY") or (sys.argv[3] if len(sys.argv) > 3 else None)
    ums_token = os.environ.get("CLINE_UMS_TOKEN") or (sys.argv[4] if len(sys.argv) > 4 else None)
    run(out_dir, message, api_key, ums_token)
