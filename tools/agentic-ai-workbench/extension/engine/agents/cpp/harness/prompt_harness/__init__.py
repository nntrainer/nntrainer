"""
Prompt Harness - Auto-generate prompt files for new CausalLM models.

Two-path approach:
1. REFERENCE PATH: If Applications/CausalLM/models/<model>/ exists with <model>_causallm.cpp
   → Extract patterns deterministically → Generate prompt
2. LLM PATH: No reference found
   → Extract HuggingFace architecture → LLM generates implementation
   → Harness validates → Extract patterns → Generate prompt
"""

from .dispatcher import PromptDispatcher
from .registry import ModelRegistry
from .prompt_validator import PromptValidator

__all__ = [
    'PromptDispatcher',
    'ModelRegistry',
    'PromptValidator',
]
