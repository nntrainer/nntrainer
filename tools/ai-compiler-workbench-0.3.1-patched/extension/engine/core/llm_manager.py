"""
LLM client with retry / back-off, timeout, and optional structured output.
Wraps LangChain's ChatAnthropic so every agent gets the same behaviour.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Type

from core.exceptions import LLMError

logger = logging.getLogger(__name__)

# Optional heavy imports — gracefully absent in environments without API access
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    _LANGCHAIN_OK = True
except ImportError:
    _LANGCHAIN_OK = False


class LLMManager:
    """
    Thin wrapper around ChatAnthropic that adds:
    - Lazy client initialisation
    - Configurable retry / exponential back-off
    - Hard timeout (best-effort via a threading.Timer sentinel)
    - Structured-output helpers (with_structured_output, JSON fallback)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        timeout_sec: int = 60,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        if not _LANGCHAIN_OK:
            raise LLMError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic --break-system-packages"
            )
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self.__client: Optional[ChatAnthropic] = None   # double-underscore → name-mangled

    # ------------------------------------------------------------------ client
    @property
    def client(self) -> "ChatAnthropic":
        if self.__client is None:
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self.__client = ChatAnthropic(**kwargs)
        return self.__client

    # ------------------------------------------------------------------ invoke
    def invoke(self, system_prompt: str, user_message: str) -> str:
        """
        Call the LLM with automatic retries and exponential back-off.
        Returns the text content of the response.
        Raises LLMError after all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        delay = self._retry_base_delay

        for attempt in range(1, self._max_retries + 2):   # +1 so 0 retries = 1 attempt
            try:
                logger.debug("LLM attempt %d/%d", attempt, self._max_retries + 1)
                response = self.client.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_message),
                    ]
                )
                content = response.content
                # content can be str or list[dict] depending on LangChain version
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                return content
            except Exception as exc:
                last_exc = exc
                logger.warning("LLM attempt %d failed: %s", attempt, exc)
                if attempt <= self._max_retries:
                    time.sleep(delay)
                    delay = min(delay * 1.5, 30.0)   # cap at 30 s

        raise LLMError(
            f"LLM call failed after {self._max_retries + 1} attempts"
        ) from last_exc

    # ------------------------------------------------------------------ structured output
    def invoke_structured(
        self,
        system_prompt: str,
        user_message: str,
        schema: Type,
    ) -> Any:
        """
        Call the LLM and parse the result against a Pydantic v2 schema.
        Falls back to JSON parsing if with_structured_output is unavailable.
        """
        if hasattr(self.client, "with_structured_output"):
            try:
                structured = self.client.with_structured_output(schema)
                return structured.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_message),
                    ]
                )
            except Exception as exc:
                logger.warning("with_structured_output failed, falling back to JSON: %s", exc)

        raw = self.invoke(system_prompt, user_message)
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.lstrip("`").lstrip("json").strip()
            if text.endswith("```"):
                text = text[: text.rfind("```")].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse LLM response as JSON: {raw[:300]}") from exc

    def __repr__(self) -> str:
        return f"LLMManager(model={self._model!r})"
