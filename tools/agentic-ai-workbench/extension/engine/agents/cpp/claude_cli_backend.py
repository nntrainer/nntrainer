"""
Claude Code CLI LLM backend for C++ generation.

Delegates LLM interactions to the local Claude Code CLI instead of using API keys.
This allows users without Anthropic/Cline API keys to still use the workbench.

The Claude Code CLI must be installed and authenticated:
    npm install -g @anthropic-ai/claude-code
    claude login

Typical usage:
    llm = _make_claude_cli_llm(state)
    response = llm(system_prompt, user_prompt)
"""
import os
import subprocess
from typing import Callable, Dict, Optional

from ..events import bus


def _make_claude_cli_llm(state: Dict) -> Optional[Callable]:
    """
    Return a callable (system, user) -> str using Claude Code CLI, or None if unavailable.

    The callable wraps subprocess calls to `claude -p` (print mode, non-interactive)
    so the caller never needs to know about the CLI -- the interface is identical
    to the Anthropic/Cline backends.

    Requires `claude` CLI to be installed and authenticated locally (`claude login`).

    Configuration keys in state:
    - claude_model: model alias/name, e.g. "sonnet", "opus", "haiku" (default: "sonnet")
    - claude_cli_timeout: timeout in seconds (default: 180)
    - claude_cli_enabled: explicitly enable/disable (default: true if CLI available)
    """
    if state.get("claude_cli_enabled") is False:
        return None

    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            bus.log(
                "Claude CLI not found or not working. "
                "Install with: npm install -g @anthropic-ai/claude-code && claude login",
                "warn"
            )
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        bus.log(
            f"Claude CLI not available ({exc}). "
            "Install with: npm install -g @anthropic-ai/claude-code",
            "warn"
        )
        return None

    model = state.get("claude_model") or os.environ.get("CLAUDE_MODEL") or "sonnet"
    timeout = int(state.get("claude_cli_timeout") or os.environ.get("CLAUDE_CLI_TIMEOUT") or 300)

    def _invoke(system: str, user: str) -> str:
        """
        Call Claude Code CLI in print mode with the given system and user prompts.

        The user prompt is piped via stdin (command-line args have length limits
        that C++ generation prompts, which embed full skeletons and layer
        catalogs, can exceed). Returns the response text, or "" on failure.
        """
        cmd = [
            "claude", "-p",
            "--model", model,
            "--system-prompt", system,
            "--output-format", "text",
            "--permission-prompts", "none",
        ]

        bus.log(f"Claude CLI: model={model}, timeout={timeout}s, "
                f"prompt_size={len(system) + len(user)} chars")

        try:
            result = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            bus.log(f"Claude CLI timed out after {timeout}s", "error")
            return ""
        except Exception as exc:
            bus.log(f"Claude CLI call failed: {exc}", "error")
            return ""

        if result.returncode != 0:
            bus.log(
                f"Claude CLI call failed (exit {result.returncode})\n"
                f"STDERR: {result.stderr[:2000]}",
                "error"
            )
            return ""

        response = result.stdout.strip()
        bus.log(f"Claude CLI response: {len(response)} chars")
        return response

    bus.log(f"C++ generation using Claude Code CLI (model: {model})")
    return _invoke
