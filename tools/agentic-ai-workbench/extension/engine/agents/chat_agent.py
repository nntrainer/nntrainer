"""Chat agent for the webview Chat panel."""
import json
import os
import sys
import subprocess

from .events import bus


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

    if not api_key and not ums_token:
        bus.chat("assistant",
                 "Chat requires an API key. Set aiCompilerWorkbench.anthropicApiKey in Settings, "
                 "or set CLINE_UMS_TOKEN environment variable. "
                 "Everything else (pipeline, graph, code, profiler) works without a key.")
        return

    if not state:
        bus.chat("assistant", "No pipeline has been run yet in this workspace -- click **Run Pipeline** first.")
        return

    if not _ensure_langchain_installed(ums_token):
        return

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        bus.chat("assistant", "Failed to import langchain after install attempt.")
        return

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
    except ImportError as e:
        if ums_token:
            bus.chat("assistant", "langchain-openai isn't installed -- run pip install langchain-openai.")
        else:
            bus.chat("assistant", "langchain-anthropic isn't installed -- run pip install langchain-anthropic.")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the orchestrator agent for an nntrainer C++ code-generation pipeline "
         "running inside a VS Code extension. Use the tools to answer questions about "
         "the most recent conversion run. Be concise and concrete."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6)

    try:
        result = executor.invoke({"input": message})
        bus.chat("assistant", result.get("output", "(no response)"))
    except Exception as exc:
        bus.chat("assistant", f"Chat request failed: {exc}")


if __name__ == "__main__":
    out_dir, message = sys.argv[1], sys.argv[2]
    api_key = os.environ.get("ANTHROPIC_API_KEY") or (sys.argv[3] if len(sys.argv) > 3 else None)
    ums_token = os.environ.get("CLINE_UMS_TOKEN") or (sys.argv[4] if len(sys.argv) > 4 else None)
    run(out_dir, message, api_key, ums_token)
