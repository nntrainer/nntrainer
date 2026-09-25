"""
Abstract base class for every agent in the pipeline.
Connects an agent to the shared event bus, the LLM manager, and the
LangChain ReAct executor so each concrete agent only has to define
get_tools(), get_system_prompt(), and run().
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.events import bus
from core.exceptions import AgentError
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Optional LangChain imports -- absent in lightweight environments
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import BaseTool
    _LC_AGENTS_OK = True
except ImportError:
    AgentExecutor = None
    create_react_agent = None
    ChatPromptTemplate = None
    BaseTool = object
    _LC_AGENTS_OK = False


class BaseAgent(ABC):
    """
    Every concrete agent inherits from this.

    Responsibilities:
    - Provides self.log() / self.status() that write to the shared event bus
      (same wire format as the original agents/events.py so the webview
       needs no changes)
    - Creates and caches a LangChain ReAct AgentExecutor on demand
    - Enforces get_tools / get_system_prompt / run contract via ABC
    """

    #: agent_id must be unique across the pipeline (used in agent_status events)
    agent_id: str = "base"

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        llm_manager:
            Shared manager instance. If None a new one is created using
            api_key.  Passing a shared instance lets multiple agents
            reuse the same underlying HTTP connection pool.
        api_key:
            Anthropic API key. Only used when llm_manager is None.
        """
        self._llm: LLMManager = llm_manager or LLMManager(api_key=api_key)
        self._executor: Optional[AgentExecutor] = None

    # ------------------------------------------------------------------ logging
    def log(self, message: str, level: str = "info") -> None:
        """Write a log line to the shared event bus (→ Logs panel in UI)."""
        bus.log(message, level)

    def status(self, status: str, detail: str = "") -> None:
        """Write an agent_status event (→ Agent Pipeline panel in UI)."""
        bus.agent_status(self.agent_id, status, detail)

    # ------------------------------------------------------------------ LangChain
    @abstractmethod
    def get_tools(self) -> List[Any]:
        """
        Return a list of LangChain BaseTool instances for the ReAct executor.
        Concrete agents that don't need LLM tooling can return [].
        """

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt injected at the top of every ReAct loop."""

    @property
    def executor(self) -> Optional[AgentExecutor]:
        """
        Lazy-initialise the ReAct AgentExecutor.
        Returns None if LangChain agent dependencies aren't installed.
        """
        if self._executor is not None:
            return self._executor

        if not _LC_AGENTS_OK:
            self.log(
                "langchain>=0.2 not fully installed -- ReAct executor unavailable; "
                "falling back to direct LLM calls.",
                "warn",
            )
            return None

        try:
            tools = self.get_tools()
            system_prompt = self.get_system_prompt()

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            agent = create_react_agent(self._llm.client, tools, prompt)
            self._executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=8,
            )
        except Exception as exc:
            raise AgentError(f"Failed to build ReAct executor for {self.agent_id}: {exc}") from exc

        return self._executor

    # ------------------------------------------------------------------ run
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary task.
        Must return a dict; at minimum {"ok": bool, "error": str | None}.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id!r})"
