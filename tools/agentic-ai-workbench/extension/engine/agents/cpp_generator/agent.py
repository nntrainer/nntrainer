"""
CppGeneratorAgent — LangChain ReAct agent that:
1. Generates nntrainer C++ from a model graph using the knowledge bank
2. Compiles the result via CppCorrector
3. If compilation fails, re-prompts the LLM with the error summary
4. Repeats up to config.cpp_generator.max_retry_attempts times
5. Streams every step to the shared event bus for the UI

Tool inventory (used by the ReAct executor):
  generate_cpp_code   — call the LLM to produce or correct code
  compile_cpp_code    — run g++ and get a success/error dict
  lookup_nntrainer_kb — search the knowledge bank for a snippet
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

try:
    from langchain_core.tools import tool as lc_tool
    _HAS_LANGCHAIN = True
except ImportError:
    lc_tool = None
    _HAS_LANGCHAIN = False

from core.base_agent import BaseAgent
from core.events import bus
from core.exceptions import AgentError, CompilationError
from core.llm_manager import LLMManager
from knowledge.nntrainer_kb import get_knowledge_bank
from .cpp_corrector import CppCorrector
from .prompts import SYSTEM_PROMPT, generation_prompt, correction_prompt

logger = logging.getLogger(__name__)


class CppGeneratorAgent(BaseAgent):
    """
    Generates and self-corrects nntrainer C++ code.
    """
    agent_id = "cpp_generator"

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(llm_manager=llm_manager, api_key=api_key)
        self._kb = get_knowledge_bank()
        self._corrector = CppCorrector()

    # ---------------------------------------------------------------- tools
    def get_tools(self) -> List[Any]:
        """Return LangChain tools for the ReAct executor."""
        agent = self            # captured in closures below
        kb    = self._kb
        corr  = self._corrector

        @lc_tool
        def generate_cpp_code(model_definition: str, error_context: str = "") -> str:
            """
            Generate (or re-generate with error context) nntrainer C++ code.
            model_definition: the model graph or .ini content.
            error_context: compiler errors from the previous attempt (empty on first call).
            """
            return agent._call_llm(model_definition, error_context, attempt=1, max_attempts=1)

        @lc_tool
        def compile_cpp_code(code: str, nntrainer_prefix: str) -> Dict[str, Any]:
            """
            Attempt to compile C++ code against a local nntrainer install.
            Returns {'success': bool, 'summary': str}.
            nntrainer_prefix: path to the nntrainer install (contains include/ and lib/).
            """
            result = corr.compile(code, nntrainer_prefix)
            return {"success": result["success"], "summary": result["summary"]}

        @lc_tool
        def lookup_nntrainer_kb(query: str) -> str:
            """
            Search the nntrainer causallm knowledge bank for API docs.
            query: what you are looking for (e.g. 'forward pass', 'layer properties').
            """
            hits = kb.search(query, top_k=3)
            return "\n\n".join(hits) if hits else "No matching docs found."

        return [generate_cpp_code, compile_cpp_code, lookup_nntrainer_kb]

    # --------------------------------------------------------- system prompt
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # --------------------------------------------------------------- run
    def run(self, model_definition: str, nntrainer_prefix: str, **_: Any) -> Dict[str, Any]:
        """
        Full generate → compile → correct loop.

        Parameters
        ----------
        model_definition:
            Model.ini content or structured graph description string.
        nntrainer_prefix:
            Path to the nntrainer install prefix (contains include/ and lib/).

        Returns
        -------
        {
          "ok":       bool,
          "code":     str,
          "attempts": int,
          "errors":   str   # last error summary if ok is False
        }
        """
        self.status("running", "Generating C++ code …")
        max_attempts = self.config.get("cpp_generator", "max_retry_attempts", 3)

        code = ""
        last_error = ""

        for attempt in range(1, max_attempts + 2):       # attempt = 1..max+1
            # ---- generate
            self.log(f"Code generation attempt {attempt}/{max_attempts + 1}")
            try:
                code = self._call_llm(
                    model_definition,
                    error_context=last_error,
                    attempt=attempt,
                    max_attempts=max_attempts + 1,
                )
            except Exception as exc:
                self.log(f"LLM call failed: {exc}", "error")
                self.status("error", str(exc))
                return {"ok": False, "code": code, "attempts": attempt, "errors": str(exc)}

            bus.code("generated_model.cpp", code)

            # ---- compile
            self.log("Compiling …")
            result = self._corrector.compile(code, nntrainer_prefix)

            if result["success"]:
                self.log("Compilation succeeded ✓")
                self.status("done", f"Code generated and compiled in {attempt} attempt(s)")
                return {"ok": True, "code": code, "attempts": attempt, "errors": ""}

            last_error = result["summary"]
            self.log(f"Compilation failed (attempt {attempt}): {last_error}", "warn")

            if attempt > max_attempts:
                # Exhausted retries — surface the last error to the user
                self.status("error", f"Compilation failed after {attempt} attempt(s)")
                return {"ok": False, "code": code, "attempts": attempt, "errors": last_error}

        # Should be unreachable, but keeps the type checker happy
        return {"ok": False, "code": code, "attempts": max_attempts + 1, "errors": last_error}

    # --------------------------------------------------------- LLM helpers
    def _call_llm(
        self,
        model_definition: str,
        error_context: str,
        attempt: int,
        max_attempts: int,
    ) -> str:
        """Build a prompt and call the LLM; extract code from the response."""
        kb_ctx = self._kb.full_context(
            char_limit=self.config.get("cpp_generator", "kb_context_limit_chars", 50_000)
        )

        if error_context:
            prompt = correction_prompt(
                model_definition=model_definition,
                previous_code=self._last_code,
                error_summary=error_context,
                kb_context=kb_ctx,
                attempt=attempt,
                max_attempts=max_attempts,
            )
        else:
            prompt = generation_prompt(model_definition, kb_ctx)

        raw = self._llm.invoke(SYSTEM_PROMPT, prompt)
        code = _extract_code(raw)
        self._last_code = code
        return code

    @property
    def _last_code(self) -> str:
        return getattr(self, "_last_code_store", "")

    @_last_code.setter
    def _last_code(self, value: str) -> None:
        self._last_code_store = value

    def __repr__(self) -> str:
        return "CppGeneratorAgent()"


# --------------------------------------------------------------- helpers
def _extract_code(response: str) -> str:
    """
    Pull C++ source from an LLM response that may or may not be wrapped
    in markdown fences.  Handles ```cpp, ```c++, and bare ``` variants.
    """
    fence_re = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence_re.search(response)
    if m:
        return m.group(1).strip()
    # No fences — assume the whole response is code
    return response.strip()
