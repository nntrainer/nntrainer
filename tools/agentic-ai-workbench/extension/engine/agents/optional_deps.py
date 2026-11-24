# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file optional_deps.py
# @brief Checks for the optional LangChain packages used by the API-key features.
# @author prachi-t17 <prachi.t@samsung.com>

"""
The optional LangChain packages (engine/requirements-agentic.txt) that the
API-key extras use: chat without the Claude CLI and Auto-Fix.

The workbench never installs packages by itself while it runs: it says which
are missing and the exact command that installs them. "Set Up / Check Python
Environment" offers the same install, with the user's consent.
"""
import importlib.util
import os
import sys
from typing import List

#: The model the API-key extras call (chat without the CLI, Auto-Fix, op
#: suggestions, the graph Focus fallback). $WORKBENCH_API_MODEL overrides it.
API_MODEL = os.environ.get("WORKBENCH_API_MODEL") or "claude-sonnet-4-6"

REQUIREMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements-agentic.txt"
)


def missing(*modules: str) -> List[str]:
    """The given top-level modules that are not importable."""
    out = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def install_hint() -> str:
    """How to install the optional packages into the interpreter running the engine."""
    return (
        f'Install them with: "{sys.executable or "python"}" -m pip install -r "{REQUIREMENTS}", '
        'or run "AI Compiler Workbench: Set Up / Check Python Environment".'
    )
