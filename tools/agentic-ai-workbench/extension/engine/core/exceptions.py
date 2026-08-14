"""
Exceptions for the AI Compiler Workbench agent pipeline.
"""


class AICompilerError(Exception):
    """Base exception."""


class AgentError(AICompilerError):
    """An agent failed to complete its task."""


class LLMError(AICompilerError):
    """LLM API call failed (timeout, parse error, auth, etc.)."""


class CompilationError(AICompilerError):
    """C++ compilation failed after all retries."""


class KnowledgeBankError(AICompilerError):
    """Knowledge bank could not be loaded or queried."""


class ConfigError(AICompilerError):
    """Configuration is invalid or missing."""
