# Claude Code CLI Backend Setup

This guide explains how to configure the agentic-ai-workbench to use Claude Code CLI for LLM-driven code generation instead of API keys.

## Why Use Claude Code CLI?

- **No API keys needed**: Works with local Claude Code CLI authentication
- **No Ollama required**: Direct access to Claude through your local CLI
- **Seamless integration**: Automatically used as a fallback when no API keys are configured
- **Organization-friendly**: For teams without API key provisioning

## Installation

### 1. Install Claude Code CLI

```bash
npm install -g @anthropic-ai/claude-cli
```

### 2. Authenticate

```bash
claude login
```

This will open a browser for you to authenticate with your Claude account. The CLI will cache your credentials locally.

### 3. Verify Installation

```bash
claude --version
claude "Hello, Claude!"
```

You should see Claude respond to your message, confirming the CLI is working.

## Configuration

The workbench will automatically use Claude Code CLI when:

1. No Cline API token is configured (`cline_ums_token`)
2. No Anthropic API key is configured (`ANTHROPIC_API_KEY`)
3. Claude Code CLI is installed and authenticated

### Via Environment Variables

Set these before running the workbench:

```bash
# Choose which model to use (default: claude-opus-4)
export CLAUDE_MODEL=claude-opus-4

# Set timeout in seconds (default: 120)
export CLAUDE_CLI_TIMEOUT=180
```

### Via Workbench Configuration

In the workbench settings (e.g., `aiCompilerWorkbench` in VSCode settings):

```json
{
  "aiCompilerWorkbench": {
    "claudeModel": "claude-opus-4",
    "claudeCliTimeout": 120,
    "claudeCliEnabled": true
  }
}
```

### Priority Order

The workbench tries LLM backends in this order:

1. **Cline/Gauss API** (if `cline_ums_token` is set)
2. **Claude Code CLI** (if `claude` command is available and authenticated)
3. **Anthropic API** (if `ANTHROPIC_API_KEY` is set)
4. **None** (skeleton generation only, no LLM calls)

## How It Works

The C++ generator delegates to Claude Code CLI through these steps:

1. **System & user prompts** are written to a temporary file
2. **subprocess call** to `claude` CLI with the prompt
3. **Response parsing** extracts the generated C++ code
4. **Temporary files** are cleaned up automatically

The entire interaction is transparent — the workbench doesn't need to know or care which backend is being used.

## Troubleshooting

### "Claude CLI not found"

```bash
# Verify installation
which claude
claude --version

# Reinstall if needed
npm install -g @anthropic-ai/claude-cli
```

### "Claude CLI not authenticated"

```bash
# Re-authenticate
claude login

# Verify with a test call
claude "Hello"
```

### "Timeout waiting for response"

The default timeout is 120 seconds. Increase it for complex models:

```bash
export CLAUDE_CLI_TIMEOUT=300
```

Or configure in workbench settings.

### "Generation succeeded but output is incomplete"

Check if the model is producing truncated responses:

1. Try a simpler model: `export CLAUDE_MODEL=claude-sonnet-4`
2. Increase timeout: `export CLAUDE_CLI_TIMEOUT=300`
3. Check workbench logs for parsing errors

### "Multiple backends available, wrong one is being used"

The workbench tries backends in order. To force Claude CLI:

1. Unset `ANTHROPIC_API_KEY`: `unset ANTHROPIC_API_KEY`
2. Remove `cline_ums_token` from settings
3. Verify CLI is available: `claude --version`

## Performance Notes

- **Cold starts**: First call may be slower as Claude CLI warms up
- **Token limits**: Responses are capped at 8192 tokens by default
- **Model selection**: 
  - `claude-opus-4` - Most capable, slower (recommended for complex models)
  - `claude-sonnet-4` - Balanced speed/quality
  - `claude-haiku-4` - Fast but may miss complex patterns

## Advanced: Custom Model Selection

To use a different Claude model:

```bash
# Via environment variable
export CLAUDE_MODEL=claude-sonnet-4

# Or in code (if you're a developer)
state["claude_model"] = "claude-haiku-4-5"
```

## Architecture Details

See [claude_cli_backend.py](extension/engine/agents/cpp/claude_cli_backend.py) for the implementation.

The backend:
- Wraps subprocess calls to the `claude` CLI
- Provides the same interface as the Anthropic backend
- Handles temporary file creation and cleanup
- Logs all interactions to the workbench event bus

## Feedback

If you encounter issues:

1. Check workbench logs (usually in the Output panel)
2. Verify Claude CLI works: `claude "test"`
3. Report issues with:
   - Error messages from logs
   - System details (`uname -a`, `node --version`, `npm --version`)
   - Configuration you're using
