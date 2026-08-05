"""
Prompt templates for the C++ Generator Agent.
All KB injection goes through NNTrainerKnowledgeBank.inject() so the
character budget is respected consistently.
"""
from __future__ import annotations

from typing import Optional


# ------------------------------------------------------------------ system
SYSTEM_PROMPT = """\
You are an expert C++ developer with deep knowledge of the nntrainer
neural-network inference library (https://github.com/nntrainer/nntrainer).

Your task is to produce complete, compilable C++ source code that:
1. Includes all required headers (<layer.h>, <model.h> etc.)
2. Uses createLayer() for every layer — never raw new
3. Constructs layers, sets their properties, then calls model->addLayer()
4. Calls model->compile() before any forward pass
5. Uses only the public nntrainer API (no internal headers)
6. Is idiomatic C++17 (auto, range-for, structured bindings where helpful)
7. Handles errors with try-catch around compile()/forward() calls
8. Emits ONLY compilable source — no prose, no markdown outside fences

When correcting code, address the exact compiler error(s) listed.
Do not rewrite sections that were not involved in the error.
"""


def generation_prompt(model_definition: str, kb_context: str) -> str:
    """
    First-pass generation prompt.
    model_definition: the model.ini content or structured graph description.
    kb_context: full nntrainer API context from the knowledge bank.
    """
    return f"""\
Generate complete, compilable C++ code for the following nntrainer model.

=== MODEL DEFINITION ===
{model_definition}
========================

Requirements:
- Complete standalone source file (main() or library header as appropriate)
- Every layer in the model definition must appear in the output
- Layer order, names, and connection topology must match exactly
- Output ONLY the C++ source wrapped in ```cpp ... ``` fences

=== nntrainer API REFERENCE ===
{kb_context}
================================
"""


def correction_prompt(
    model_definition: str,
    previous_code: str,
    error_summary: str,
    kb_context: str,
    attempt: int,
    max_attempts: int,
) -> str:
    """
    Self-correction prompt: include the previous code and the compiler errors.
    """
    return f"""\
The previously generated C++ code failed to compile (attempt {attempt}/{max_attempts}).
Fix ONLY the errors listed below. Do not restructure unrelated code.

=== COMPILER ERRORS ===
{error_summary}
=======================

=== PREVIOUS CODE ===
```cpp
{previous_code}
```
======================

=== MODEL DEFINITION (for reference) ===
{model_definition}
=========================================

=== nntrainer API REFERENCE ===
{kb_context}
================================

Output ONLY the corrected C++ source in ```cpp ... ``` fences.
"""
