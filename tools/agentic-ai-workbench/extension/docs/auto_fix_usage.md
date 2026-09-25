# Auto-Fix Agent - Usage Guide

## Overview

The Auto-Fix Agent automatically fixes C++ compilation errors in nntrainer-generated code using LLM (Anthropic Claude or Cline).

## Features

### 1. **Cline API Support**
Use Cline (VS Code extension) as the LLM provider instead of Anthropic API.

### 2. **Error Categorization**
Automatically identifies fixable vs unfixable errors:
- **Fixable**: Missing declarations, syntax errors, wrong signatures, type mismatches
- **Unfixable**: Missing system headers, linker errors, missing libraries

### 3. **Rule-Based Corrections**
Applies quick fixes before calling LLM:
- Add missing `#include` directives
- Add `using namespace std;`
- Fix `main()` return type
- Add `return 0;` to main

### 4. **Configurable Max Iterations**
Control how many auto-fix attempts are made (default: 2).

### 5. **Fix History Learning**
Tracks past fix attempts for improved future fixes.

---

## Configuration

### Environment Variables

```bash
# Max fix iterations
export AUTO_FIX_MAX_ITERATIONS=3

# Enable Cline mode
export AUTO_FIX_USE_CLINE=true

# Cline configuration
export CLINE_API_BASE="http://localhost:6543/v1"
export CLINE_API_KEY="dummy"
export CLINE_MODEL="claude-sonnet-4-5-20250929"

# Anthropic configuration (if not using Cline)
export AUTO_FIX_MODEL="claude-sonnet-4-6"
export AUTO_FIX_MAX_TOKENS=4000
```

### Programmatic Configuration

```python
from engine.agents.orchestrator import run_pipeline

final_state = run_pipeline(
    model_name="meta-llama/Llama-2-7b",
    out_dir="./nntrainer_out",
    api_key=None,  # Not needed if using Cline
    
    # Auto-fix configuration
    max_fix_iterations=3,
    auto_fix_use_cline=True,
    cline_api_base="http://localhost:6543/v1",
    cline_api_key="dummy",
    cline_model="claude-sonnet-4-5-20250929",
    auto_fix_model="claude-sonnet-4-6",
    auto_fix_max_tokens=4000,
)
```

---

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Virtual environment** (recommended)
3. **Cline VS Code extension** (if using Cline mode)

### Install Dependencies

```bash
cd /storage_data/snap/Prachi/nntrainer/tools/agentic-ai-workbench/extension/engine

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check if langchain-openai is installed (for Cline support)
python -c "from langchain_openai import ChatOpenAI; print('OK')"

# Check if langchain-anthropic is installed
python -c "from langchain_anthropic import ChatAnthropic; print('OK')"
```

---

## Usage Examples

### Example 1: Using Cline API

```python
from engine.agents.orchestrator import run_pipeline

# Make sure Cline extension is running in VS Code
final_state = run_pipeline(
    model_name="facebook/opt-125m",
    out_dir="./nntrainer_out",
    auto_fix_use_cline=True,
    cline_api_base="http://localhost:6543/v1",
    cline_api_key="dummy",
    cline_model="claude-sonnet-4-5-20250929",
    max_fix_iterations=3,
)
```

### Example 2: Using Anthropic API

```python
from engine.agents.orchestrator import run_pipeline

final_state = run_pipeline(
    model_name="facebook/opt-125m",
    out_dir="./nntrainer_out",
    api_key="sk-ant-xxx",  # Your Anthropic API key
    auto_fix_use_cline=False,
    auto_fix_model="claude-sonnet-4-6",
    max_fix_iterations=2,
)
```

### Example 3: Using Environment Variables

```bash
# Set environment variables
export AUTO_FIX_USE_CLINE=true
export CLINE_API_BASE="http://localhost:6543/v1"
export AUTO_FIX_MAX_ITERATIONS=3

# Run pipeline (uses env vars automatically)
python -c "
from engine.agents.orchestrator import run_pipeline
run_pipeline('facebook/opt-125m', './nntrainer_out')
"
```

---

## Architecture

### Auto-Fix Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run(state)                                            │
│                                                                              │
│  1. Check prerequisites:                                                     │
│     └─► api_key OR auto_fix_use_cline                                        │
│     └─► cpp_code AND compile_log must exist                                  │
│     └─► fix_iterations < max_fix_iterations                                  │
│                                                                              │
│  2. Categorize errors:                                                       │
│     ├─► Fixable: declaration, syntax, signature errors                       │
│     └─► Unfixable: missing headers, linker errors                            │
│                                                                              │
│  3. Apply rule-based corrections:                                            │
│     ├─► Add missing includes                                                 │
│     ├─► Add using namespace std                                              │
│     └─► Fix main() return type                                               │
│                                                                              │
│  4. Build optimized prompt:                                                  │
│     ├─► Error summary with types                                             │
│     ├─► Unfixable error warnings                                             │
│     ├─► Fix history context                                                  │
│     └─► Common nntrainer issues reference                                    │
│                                                                              │
│  5. Get LLM client:                                                          │
│     ├─► Cline: ChatOpenAI(api_base=cline_api_base)                           │
│     └─► Anthropic: ChatAnthropic(model=auto_fix_model)                       │
│                                                                              │
│  6. Call LLM and process response:                                           │
│     ├─► Strip markdown fences                                                │
│     ├─► Update state["cpp_code"]                                             │
│     ├─► Write to file                                                        │
│     └─► Record fix history                                                   │
│                                                                              │
│  7. Return state ──► Triggers recompile via LangGraph edge                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Patterns

### Fixable Error Patterns

| Pattern | Error Type |
|---------|------------|
| `was not declared in this scope` | missing_declaration |
| `expected.*before.*token` | syntax_error |
| `no matching function for call` | wrong_signature |
| `no member named.*in` | missing_member |
| `use of undeclared identifier` | undeclared_identifier |
| `cannot convert.*to.*in assignment` | type_mismatch |
| `invalid conversion.*to.*from` | invalid_conversion |
| `too few arguments to function` | missing_arguments |
| `too many arguments to function` | extra_arguments |
| `class.*has no member named` | missing_class_member |
| `expected.*at end of input` | missing_brace |

### Unfixable Error Patterns

| Pattern | Error Type | Action |
|---------|------------|--------|
| `fatal error:.*No such file or directory` | missing_header_file | Skip, notify user |
| `undefined reference to` | linker_error | Skip, notify user |
| `cannot find -l` | missing_library | Skip, notify user |
| `ld returned 1 exit status` | linker_failure | Skip, notify user |
| `nntrainer not found` | nntrainer_not_installed | Skip, notify user |

---

## Troubleshooting

### Issue: "No API key available"

**Solution:** Either provide an Anthropic API key or enable Cline mode:

```python
# Option 1: Use Anthropic
run_pipeline(..., api_key="sk-ant-xxx")

# Option 2: Use Cline
run_pipeline(..., auto_fix_use_cline=True)
```

### Issue: "langchain-openai not installed"

**Solution:** Install the package:

```bash
source venv/bin/activate
pip install langchain-openai
```

### Issue: Cline connection refused

**Solution:** Make sure Cline extension is running in VS Code and the API base is correct:

```python
run_pipeline(
    ...,
    auto_fix_use_cline=True,
    cline_api_base="http://localhost:6543/v1",  # Default Cline port
)
```

### Issue: Max iterations reached without success

**Solution:** 
1. Check if errors are unfixable (missing headers, linker errors)
2. Increase max iterations: `max_fix_iterations=5`
3. Try a different model: `cline_model="claude-3-opus-20240229"`

---

## Files Modified

| File | Purpose |
|------|---------|
| `agents/state.py` | Added auto-fix configuration fields |
| `agents/auto_fix.py` | Complete rewrite with all improvements |
| `agents/orchestrator.py` | Updated with configuration support |
| `requirements.txt` | Added langchain-openai |

---

## API Reference

### `run_pipeline()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | - | HuggingFace model name |
| `out_dir` | str | - | Output directory |
| `api_key` | str | None | Anthropic API key |
| `max_fix_iterations` | int | 2 | Max auto-fix attempts |
| `auto_fix_use_cline` | bool | False | Use Cline API |
| `cline_api_base` | str | `http://localhost:6543/v1` | Cline API URL |
| `cline_api_key` | str | `dummy` | Cline API key |
| `cline_model` | str | `claude-sonnet-4-5-20250929` | Cline model |
| `auto_fix_model` | str | `claude-sonnet-4-6` | Anthropic model |
| `auto_fix_max_tokens` | int | 4000 | Max LLM tokens |

---

## License

Apache 2.0 - Same as nntrainer project.
