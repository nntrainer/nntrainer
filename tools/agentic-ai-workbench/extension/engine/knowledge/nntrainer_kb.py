"""
nntrainer causallm knowledge bank.
Ships pre-built as knowledge/nntrainer_api.json.
Used by CppGeneratorAgent to inject correct API context into prompts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.exceptions import KnowledgeBankError
from config import get_config

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "nntrainer_api.json"


class NNTrainerKnowledgeBank:
    """
    Loads the pre-built nntrainer causallm API reference and exposes
    helpers that the CppGeneratorAgent uses to build LLM prompts.

    The JSON schema expected in nntrainer_api.json:
    {
      "headers": { "<filename>": "<full header content>", ... },
      "examples": [ { "pattern": str, "code": str, "note": str }, ... ],
      "notes":    [ "<string>", ... ]
    }
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _KB_PATH
        self._data: Dict[str, Any] = {"headers": {}, "examples": [], "notes": []}
        self._loaded = False
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Knowledge bank not found at %s", self._path)
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                self._data = json.load(fh)
            self._loaded = True
            logger.debug(
                "Knowledge bank loaded: %d headers, %d examples",
                len(self._data.get("headers", {})),
                len(self._data.get("examples", [])),
            )
        except Exception as exc:
            raise KnowledgeBankError(f"Failed to load KB from {self._path}: {exc}") from exc

    # ------------------------------------------------------------------ context
    def full_context(self, char_limit: Optional[int] = None) -> str:
        """
        Return the entire KB formatted as Markdown for injection into an LLM prompt.
        char_limit is taken from config if not supplied.
        """
        if char_limit is None:
            char_limit = get_config().get("cpp_generator", "kb_context_limit_chars", 50_000)

        parts: List[str] = ["# nntrainer causallm API Reference\n"]

        # --- Headers ---
        parts.append("## Header files\n")
        for name, content in sorted(self._data.get("headers", {}).items()):
            parts.append(f"### {name}\n```cpp\n{content[:3_000]}\n```\n")

        # --- Examples ---
        parts.append("## Usage examples\n")
        for ex in self._data.get("examples", []):
            parts.append(f"```cpp\n{ex.get('code', '')}\n```")
            if ex.get("note"):
                parts.append(f"*{ex['note']}*\n")

        # --- Notes ---
        parts.append("## Notes\n")
        for note in self._data.get("notes", []):
            parts.append(f"- {note}")

        ctx = "\n".join(parts)
        if len(ctx) > char_limit:
            ctx = ctx[:char_limit] + "\n… (truncated)"
        return ctx

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Keyword search across headers and examples.
        Returns up to top_k matching snippets.
        """
        q = query.lower()
        hits: List[str] = []

        for name, content in self._data.get("headers", {}).items():
            if q in name.lower() or q in content.lower():
                hits.append(f"### {name}\n```cpp\n{content[:800]}\n```")

        for ex in self._data.get("examples", []):
            text = ex.get("code", "") + ex.get("note", "") + ex.get("pattern", "")
            if q in text.lower():
                hits.append(f"```cpp\n{ex['code'][:400]}\n```\n*{ex.get('note', '')}*")

        return hits[:top_k]

    # ------------------------------------------------------------------ inject
    def inject(self, prompt: str, query: str = "") -> str:
        """
        Append knowledge bank context to prompt.
        Uses search() when query given, full_context() otherwise.
        """
        if query:
            snippets = self.search(query, top_k=5)
            ctx = "\n\n".join(snippets) if snippets else "(no matching docs found)"
        else:
            ctx = self.full_context()
        return f"{prompt}\n\n---\n## nntrainer API context\n{ctx}"

    # ------------------------------------------------------------------ misc
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"NNTrainerKnowledgeBank("
            f"headers={len(self._data.get('headers', {}))}, "
            f"examples={len(self._data.get('examples', []))})"
        )


# ---------------------------------------------------------------- singleton
_kb: Optional[NNTrainerKnowledgeBank] = None


def get_knowledge_bank() -> NNTrainerKnowledgeBank:
    global _kb
    if _kb is None:
        _kb = NNTrainerKnowledgeBank()
    return _kb
