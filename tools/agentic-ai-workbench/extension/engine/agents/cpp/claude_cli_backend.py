# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file claude_cli_backend.py
# @brief Claude Code CLI backend: runs claude -p as a plain text generator.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Claude Code CLI backend: runs `claude -p` (print mode) as a plain text
generator for the workbench's C++ generation, review and chat.

The Claude Code CLI must be installed and signed in (`claude` once
interactively). How the executable is found, in order:
  1. an explicit path (aiCompilerWorkbench.claudePath -> $CLAUDE_PATH)
  2. `claude` on PATH (shutil.which honours PATHEXT, so an npm-installed
     `claude.cmd` on Windows is found too)
  3. the native installer's ~/.local/bin/claude(.exe)
  4. the copy bundled with the Claude Code extension for VS Code / Cursor

How it is called:
  * The whole prompt goes through stdin. Only a one-line system prompt is
    passed on the command line, so long prompts never hit the Windows
    command-line limit (32K, or 8K through a .cmd wrapper).
  * Tools are switched off (`--tools ""`), sessions are not saved, MCP servers
    and slash commands are not loaded -- each call is a pure text request.
    Flags are only passed if this CLI's `--help` lists them.
  * It runs in an empty temporary folder, so no CLAUDE.md or project settings
    from wherever the workbench happens to run leak into the request.
  * Input and output are UTF-8 on every platform.

Typical use:
    cli = ClaudeCLI()                         # raises ClaudeCLIError if not found
    text = cli.run(system_prompt, user_prompt)
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Union

try:
    from ..events import bus as BUS
except ImportError:  # loaded on its own (the tests do that), without the agents package
    BUS = None

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 900
SHORT_SYSTEM = (
    "You are a code generator for nntrainer's CausalLM application. Follow the instructions "
    "in the message exactly and reply only in the output format it asks for."
)

_HELP_CACHE: Dict[str, str] = {}


class ClaudeCLIError(RuntimeError):
    """The Claude CLI is missing or a call failed."""


def _log(msg: str, level: str = "info") -> None:
    if BUS is not None:
        BUS.log(msg, level)


def _version_key(path: str):
    match = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in match.groups()) if match else (0, 0, 0)


def find_claude(explicit: Optional[str] = None) -> Optional[str]:
    """Path of the Claude Code CLI, or None when it cannot be found."""
    for cand in (explicit, os.environ.get("CLAUDE_PATH")):
        if cand:
            found = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
            if found:
                return found
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    for cand in (
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(home, ".claude", "local", "claude"),
    ):
        if os.path.isfile(cand):
            return cand
    bundled = []
    for editor in (".vscode", ".vscode-insiders", ".cursor", ".windsurf"):
        bundled += glob.glob(os.path.join(home, editor, "extensions", "anthropic.claude-code-*",
                                          "resources", "native-binary", "claude*"))
    bundled = [p for p in bundled if os.path.basename(p).lower() in ("claude", "claude.exe")]
    if bundled:
        return max(bundled, key=_version_key)
    return None


def _prefix(exe: Union[str, Sequence[str]]) -> List[str]:
    if not isinstance(exe, str):
        return list(exe)                          # an explicit command prefix (tests)
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", exe]
    return [exe]


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    # Running inside a Claude Code session sets these; a nested CLI would treat
    # itself as part of that session.
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class ClaudeCLI:
    """Runs `claude -p` with the prompt on stdin and returns the reply text."""
    def __init__(self, exe: Union[None, str, Sequence[str]] = None, model: Optional[str] = None,
                 timeout: Optional[int] = None):
        if exe is None or isinstance(exe, str):
            path = find_claude(exe)
            if not path:
                raise ClaudeCLIError(
                    "Claude Code CLI not found. Install it (https://docs.claude.com/claude-code),"
                    " sign in once by running `claude`, or set aiCompilerWorkbench.claudePath to"
                    " the executable."
                )
            self.exe: Union[str, List[str]] = path
        else:
            self.exe = list(exe)
        self.model = model or os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
        self.timeout = int(timeout or os.environ.get("CLAUDE_CLI_TIMEOUT") or DEFAULT_TIMEOUT)
        self._help = self._read_help()

    def _read_help(self) -> str:
        key = " ".join(_prefix(self.exe))
        if key not in _HELP_CACHE:
            try:
                proc = subprocess.run(
                    _prefix(self.exe) + ["--help"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    env=_child_env(),
                    check=False,
                )
                _HELP_CACHE[key] = (proc.stdout or "") + (proc.stderr or "")
            except (OSError, subprocess.SubprocessError) as exc:
                raise ClaudeCLIError(f"could not run {key} --help: {exc}") from exc
        return _HELP_CACHE[key]

    def supports(self, flag: str) -> bool:
        """True when this CLI version lists `flag` in its --help."""
        return re.search(re.escape(flag) + r"(?![\w-])", self._help) is not None

    def command(self) -> List[str]:
        """The command line for one request: print mode, text output, no tools."""
        cmd = _prefix(self.exe) + ["-p", "--output-format", "text", "--model", self.model]
        if self.supports("--tools"):
            cmd += ["--tools", ""]
        for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
            if self.supports(flag):
                cmd.append(flag)
        if self.supports("--system-prompt"):
            cmd += ["--system-prompt", SHORT_SYSTEM]
        return cmd

    def run(self, system: str, user: str) -> str:
        """Send `system` + `user` (both through stdin) and return the reply.
        Raises ClaudeCLIError on any failure."""
        prompt = f"<instructions>\n{system.strip()}\n</instructions>\n\n{user}" if system else user
        cwd = tempfile.mkdtemp(prefix="workbench_claude_")
        try:
            proc = subprocess.run(
                self.command(),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=cwd,
                env=_child_env(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"Claude CLI timed out after {self.timeout}s "
                                 "(raise aiCompilerWorkbench.claudeCliTimeout)") from exc
        except OSError as exc:
            raise ClaudeCLIError(f"could not start the Claude CLI ({self.exe}): {exc}") from exc
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            detail = (proc.stderr or out or "").strip()[-2000:]
            raise ClaudeCLIError(f"Claude CLI exited with {proc.returncode}: {detail}")
        if not out:
            raise ClaudeCLIError(
                "Claude CLI returned an empty reply"
                + (f": {proc.stderr.strip()[-500:]}" if proc.stderr else "")
            )
        return out

    def describe(self) -> str:
        """The CLI, the model and the timeout, for the log."""
        return f"{' '.join(_prefix(self.exe))} (model {self.model}, timeout {self.timeout}s)"
