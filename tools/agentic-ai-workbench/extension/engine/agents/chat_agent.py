# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file chat_agent.py
# @brief Chat agent for the webview Chat panel.
# @author prachi-t17 <prachi.t@samsung.com>

"""Chat agent for the webview Chat panel."""
import json
import os
import sys

from knowledge import codegen as cg

from .cpp.claude_cli_backend import ClaudeCLI, ClaudeCLIError
from .events import bus
from .optional_deps import API_MODEL, install_hint, missing


def _load_state(out_dir: str) -> dict:
    state_path = os.path.join(out_dir, "state.json")
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Chat through the Claude Code CLI (default): no API key needed. Each question
# goes to `claude -p` with the latest run's state as context and the recent
# conversation (kept in <out_dir>/chat_history.json); tools are off, so the
# chat explains and advises but never edits files.
# ---------------------------------------------------------------------------
CHAT_SYSTEM = (
    "You are the assistant inside the AI Compiler Workbench, a VS Code extension that converts"
    " Hugging Face models into nntrainer CausalLM C++ (nntrainer's Applications/CausalLM). The"
    " state of the latest run is given below. Answer the user's question concisely and concretely,"
    " citing parts, files, statuses and errors from that context. You cannot change files or run"
    " anything: when something should change, say exactly what, and point to the workbench action"
    " that does it (Review tab: Approve / Edit in editor + Re-check edits / Regenerate with a"
    " comment / Compile / Fix with Claude; or Run Pipeline). If the context does not contain the"
    " answer, say so instead of guessing. Plain text; short code snippets are fine."
)
HISTORY_FILE = "chat_history.json"
MAX_TURNS = 8


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"\n... ({len(text) - limit} more characters)"
    )


# pylint: disable-next=too-many-locals,too-many-branches
def _run_context(state: dict) -> str:
    if not state:
        return "No pipeline has run yet in this workspace."
    cfg = state.get("hf_config") or {}
    keys = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "tie_word_embeddings",
        "rope_theta",
        "rope_scaling",
        "sliding_window",
    )
    parts = [
        f"Model: {state.get('model_name')} ({state.get('architecture')})",
        "Config: " + json.dumps({k: cfg.get(k) for k in keys if k in cfg}, default=str),
        "Compatibility: " + json.dumps((state.get("report") or {}).get("summary", {}), default=str),
    ]
    if state.get("errors"):
        parts.append("Pipeline errors: " + json.dumps(state["errors"][-6:], default=str))
    nntrainer_graph_view = state.get("nntrainer_graph_view") or {}
    if nntrainer_graph_view.get("nodes"):
        types = {}
        for node in nntrainer_graph_view["nodes"]:
            types[node.get("type", "?")] = types.get(node.get("type", "?"), 0) + 1
        parts.append(
            f"nntrainer graph: {len(nntrainer_graph_view['nodes'])} nodes, by type"
            f" {json.dumps(types)}"
        )
    review_path = state.get("codegen_review_path")
    review = None
    if review_path and os.path.exists(review_path):
        try:
            review = cg.load_review(review_path)            # fills in the compile state
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ):  # an unusual review: show it as stored
            with open(review_path, encoding="utf-8") as handle:
                review = json.load(handle)
    if review:
        parts.append(
            f"Generated code: {review.get('out_dir')} (generator: {review.get('generator')}, "
            f"checks passed: {review['validation']['ok']}, attempts: {review.get('attempts')})"
        )
        for part in review["parts"]:
            line = f"- {part['hook']}: {part['action']} -> {part['status']}"
            if part.get("source_entry"):
                line += f" (code from {part['source_entry']})"
            if part.get("evidence"):
                line += " because " + "; ".join(part["evidence"][:4])
            parts.append(line)
        if review["validation"].get("errors"):
            parts.append("Failed checks: " + "; ".join(review["validation"]["errors"][:10]))
        ready = review.get("compile_ready") or {}
        compile_res = review.get("compile") or {}
        parts.append(
            f"Compile: {compile_res.get('status', 'not_run')}; ready:"
            f" {ready.get('ok')} ({ready.get('why', '')})"
        )
        for error in (compile_res.get("errors") or [])[:10]:
            parts.append(f"  compile error {error['file']}:{error['line']}: {error['message']}")
        for name, path in sorted(review.get("files", {}).items()):
            if (
                name.endswith((".h", ".cpp"))
                and not name.startswith("register_")
                and os.path.exists(path)
            ):
                with open(path, encoding="utf-8") as handle:
                    parts.append(f"=== {name} ===\n" + _clip(handle.read(), 14000))
    prof = state.get("profile") or {}
    if prof:
        parts.append(
            "Profile: "
            + json.dumps({k: v for k, v in prof.items() if k != "layers"}, default=str)[:600]
        )
    return "\n".join(parts)


def _load_history(out_dir: str) -> list:
    try:
        with open(os.path.join(out_dir, HISTORY_FILE), encoding="utf-8") as handle:
            return json.load(handle)[-MAX_TURNS:]
    except (OSError, ValueError):
        return []


def _save_history(out_dir: str, history: list) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, HISTORY_FILE), "w", encoding="utf-8") as handle:
            json.dump(history[-MAX_TURNS:], handle, indent=1)
    except OSError:
        pass


def _ask(out_dir: str, message: str, state: dict, answer) -> None:
    """Send the question with the latest run as context and the recent
    conversation; `answer(system, user)` returns the reply."""
    history = _load_history(out_dir)
    convo = "\n".join(f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history)
    user = (f"# Latest run\n{_run_context(state)}\n\n"
            + (f"# Conversation so far\n{_clip(convo, 12000)}\n\n" if convo else "")
            + f"# Question\n{message}")
    reply = answer(CHAT_SYSTEM, user)
    bus.chat("assistant", reply)
    _save_history(out_dir, history + [{"user": message, "assistant": reply}])


def run_claude_cli(out_dir: str, message: str, state: dict, cli=None) -> bool:
    """Answer with the Claude CLI. False when the CLI is not available."""
    if cli is None:
        try:
            cli = ClaudeCLI(
                os.environ.get("CLAUDE_PATH") or None,
                os.environ.get("CLAUDE_MODEL") or None,
                int(os.environ.get("CLAUDE_CHAT_TIMEOUT") or 300),
            )
        except ClaudeCLIError:
            return False
    try:
        _ask(out_dir, message, state, cli.run)
    except ClaudeCLIError as exc:
        bus.chat("assistant", f"The Claude CLI could not answer: {exc}")
    return True


def run_api(out_dir: str, message: str, state: dict, api_key: str, llm=None) -> None:
    """Answer with an Anthropic API key: the same prompt as the CLI chat, in one
    model call (langchain-anthropic; `llm` replaces the model in tests)."""
    if llm is None:
        # pylint: disable-next=import-outside-toplevel,import-error  # optional package
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=API_MODEL, max_tokens=1000, api_key=api_key)

    def answer(system: str, user: str) -> str:
        content = llm.invoke([("system", system), ("human", user)]).content
        if isinstance(content, list):                    # a list of content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return content or "(no response)"

    try:
        _ask(out_dir, message, state, answer)
    # network, quota, a bad key: say so in the chat
    except Exception as exc:  # pylint: disable=broad-except
        bus.chat("assistant", f"Chat request failed: {exc}")


def run(out_dir: str, message: str, api_key: str = None):
    """Answer one chat message with the Claude CLI or, when a key is stored, the Anthropic API."""
    state = _load_state(out_dir)

    if os.environ.get("WORKBENCH_CHAT_BACKEND", "claude-cli") != "api":
        if run_claude_cli(out_dir, message, state):
            return

    if not api_key:
        bus.chat(
            "assistant",
            "Chat uses the Claude Code CLI, which was not found. Install it and sign in once by"
            " running `claude` (or set aiCompilerWorkbench.claudePath). Alternatively store an"
            " Anthropic API key with **AI Compiler Workbench: Set Anthropic API Key**.",
        )
        return

    gone = missing("langchain_anthropic", "langchain_core")
    if gone:
        bus.chat("assistant", f"Chat with an API key needs the optional LangChain packages "
                              f"({', '.join(gone)} not installed). " + install_hint())
        return
    run_api(out_dir, message, state, api_key)


def main(argv) -> None:
    """`python -m agents.chat_agent <out_dir> <message>`, as the extension runs it."""
    # The key comes only from the environment: a command-line argument would be
    # visible to other users in the process list.
    run(argv[1], argv[2], os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    main(sys.argv)
