"""
Chat agent backing the webview's "Chat" panel.

A small LangChain tool-calling agent that answers free-form questions
about the *last completed pipeline run* for this workspace (loaded
from <out_dir>/state.json -- see orchestrator.run_pipeline). It does
not re-run the pipeline itself; "Run Pipeline" in the toolbar is a
separate, explicit action. This keeps the chat cheap (no re-tracing a
model just to answer "how many layers did that have?") and keeps the
Orchestrator's own pipeline run as the one authoritative, explicit
action a person takes.
"""
import json
import os
import sys

from .events import bus


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

    return [get_model_summary, list_unsupported_ops, get_artifacts, get_profile]


def run(out_dir: str, message: str, api_key: str):
    state = _load_state(out_dir)

    if not api_key:
        bus.chat("assistant",
                 "Set aiCompilerWorkbench.anthropicApiKey in Settings to enable chat. "
                 "Everything else (pipeline, graph, code, profiler) works without it.")
        return

    if not state:
        bus.chat("assistant", "No pipeline has been run yet in this workspace -- click **Run Pipeline** first.")
        return

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        bus.chat("assistant", "langchain / langchain-anthropic aren't installed -- run pip install -r requirements.txt.")
        return

    tools = _make_tools(state)
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=600, api_key=api_key)
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
    # Key comes from the environment (not argv, so it can't leak via process
    # listings); tolerate a positional arg for backward compatibility.
    api_key = os.environ.get("ANTHROPIC_API_KEY") or (sys.argv[3] if len(sys.argv) > 3 else None)
    run(out_dir, message, api_key)
