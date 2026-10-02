# Claude Code CLI Integration Guide

This document explains how the Claude Code CLI backend integrates into the workbench LLM infrastructure and how to use it in other modules.

## Architecture

The integration follows a **provider-agnostic** pattern:

```
┌─────────────────────────────────────────┐
│    C++ Generation Code                  │
│  (llm_codegen.py, llm_generator.py)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  _make_llm()        │
        │  (backend selector) │
        └────┬─────────────────┘
             │
    ┌────────┴────────┬────────────────┐
    ▼                 ▼                ▼
┌───────────┐  ┌──────────────┐  ┌──────────────┐
│ Cline API │  │Claude CLI    │  │Anthropic API │
└───────────┘  │(subprocess)  │  └──────────────┘
               └──────────────┘

Returns: Callable[[str, str], str]
         (system: str, user: str) -> response: str
```

## Using Claude CLI Backend

### For C++ Generation

The backend is automatically integrated. Just ensure Claude CLI is installed and authenticated:

```bash
# Already handled in llm_codegen.py:
llm = _make_llm(state)  # Returns CLI backend if available
response = llm(system_prompt, user_prompt)  # Works with any backend
```

### For Other LLM Tasks

To add Claude CLI support to other agents:

```python
from extension.engine.agents.cpp.claude_cli_backend import _make_claude_cli_llm
from extension.engine.agents.events import bus

# Create an LLM backend
state = {"claude_model": "claude-opus-4"}
llm = _make_claude_cli_llm(state)

if llm is None:
    bus.log("Claude CLI not available", "warn")
    return

# Use it
response = llm(system_prompt, user_prompt)
```

### Configuring via State

The backend respects configuration in the `state` dict:

```python
state = {
    # Optional: which model to use
    "claude_model": "claude-sonnet-4",
    
    # Optional: timeout in seconds
    "claude_cli_timeout": 180,
    
    # Optional: explicitly disable Claude CLI
    "claude_cli_enabled": False,
}

from extension.engine.agents.cpp.claude_cli_backend import _make_claude_cli_llm
llm = _make_claude_cli_llm(state)
```

### Fallback Patterns

Recommended pattern for trying multiple backends:

```python
def get_llm_backend(state: Dict):
    """Try backends in order: Cline > Claude CLI > Anthropic."""
    
    # Try Cline
    if state.get("cline_ums_token"):
        # ... create Cline backend
        pass
    
    # Try Claude CLI
    from extension.engine.agents.cpp.claude_cli_backend import _make_claude_cli_llm
    llm = _make_claude_cli_llm(state)
    if llm:
        return llm
    
    # Try Anthropic
    if state.get("anthropic_api_key"):
        # ... create Anthropic backend
        pass
    
    return None
```

## Implementation Details

### File Structure

```
extension/engine/agents/cpp/
├── llm_codegen.py              # Main C++ generation pipeline
│   └── _make_llm()             # Backend selector
├── claude_cli_backend.py        # Claude CLI implementation
└── llm_generator.py            # Alternative generator using CLI
```

### Backend Interface

All backends must return a callable with this signature:

```python
def llm_backend(system: str, user: str) -> str:
    """
    Generate a response using the LLM.
    
    Args:
        system: System prompt (role, context, constraints)
        user: User prompt (actual request)
    
    Returns:
        str: The LLM's response text, or empty string on failure
    """
```

### Error Handling

The Claude CLI backend:

1. **Gracefully handles missing CLI**: Returns `None` from `_make_claude_cli_llm()`
2. **Logs all failures**: Via `bus.log()` so they appear in workbench logs
3. **Times out reliably**: Using subprocess timeout (default 120s)
4. **Cleans up temp files**: Even on exceptions
5. **Never raises**: Always returns empty string on failure

## Configuration Precedence

The backend reads configuration from multiple sources:

1. **state dict** (highest priority, passed directly)
2. **Environment variables** (e.g., `CLAUDE_MODEL`)
3. **Hard defaults** (e.g., "claude-opus-4")

Example:
```python
state = {"claude_model": "claude-sonnet-4"}
# Uses "claude-sonnet-4" from state

state = {}  # No model specified
os.environ["CLAUDE_MODEL"] = "claude-haiku-4-5"
# Uses "claude-haiku-4-5" from environment

# Uses "claude-opus-4" (hard default)
```

## Extending the Backend

### Custom Prompt Formatting

If you need special prompt formatting, subclass or wrap:

```python
from extension.engine.agents.cpp.claude_cli_backend import _make_claude_cli_llm

def _make_custom_llm(state: Dict):
    """Claude CLI with custom prompt preprocessing."""
    base_llm = _make_claude_cli_llm(state)
    if base_llm is None:
        return None
    
    def custom_invoke(system: str, user: str) -> str:
        # Transform prompts before sending
        system = f"[CUSTOM SYSTEM]\n{system}\n[END]"
        user = f"[CUSTOM USER]\n{user}\n[END]"
        return base_llm(system, user)
    
    return custom_invoke
```

### Caching Responses

The CLI backend doesn't cache. If needed, add at a higher level:

```python
from functools import lru_cache

base_llm = _make_claude_cli_llm(state)

@lru_cache(maxsize=128)
def cached_llm(system: str, user: str) -> str:
    return base_llm(system, user)
```

**Note**: Caching for code generation is often undesirable since prompts include unique file paths, configs, etc. Be selective about what you cache.

## Debugging

### Enable Debug Logging

The backend logs to `bus.log()`, which appears in the workbench Output panel.

### Test the CLI Directly

```bash
# Verify CLI works
claude --version

# Test with a simple prompt
echo "Say hello" | claude

# Test with a file
claude < my_prompt.txt
```

### Check Integration in Code

```python
from extension.engine.agents.cpp.claude_cli_backend import _make_claude_cli_llm
llm = _make_claude_cli_llm({})

if llm is None:
    print("Claude CLI backend not available")
else:
    response = llm("You are helpful.", "Say hello.")
    print(response)
```

## Performance Characteristics

| Aspect | Details |
|--------|---------|
| **Latency** | ~5-15s for typical C++ prompts (warm CLI) |
| **Throughput** | Sequential (one prompt at a time) |
| **Token limit** | 8192 output tokens (configurable) |
| **Retry behavior** | No automatic retries; handled by caller |
| **Memory** | Minimal; temp files cleaned up automatically |

## Migration from API Keys

If you're migrating from API-key-based LLMs:

1. **Remove API keys**: Unset `ANTHROPIC_API_KEY`, `CLINE_UMS_TOKEN`
2. **Install Claude CLI**: `npm install -g @anthropic-ai/claude-cli`
3. **Authenticate**: `claude login`
4. **No code changes needed**: The workbench automatically detects and uses the CLI

The existing code is already set up to try Claude CLI as the second-priority backend.

## Support and Troubleshooting

See [CLAUDE_CLI_SETUP.md](CLAUDE_CLI_SETUP.md) for user-facing setup and troubleshooting.

For development issues, check:

1. Workbench logs (Output panel)
2. Claude CLI directly: `claude --version`
3. Backend availability: `_make_claude_cli_llm({}) is not None`
