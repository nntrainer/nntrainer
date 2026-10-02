# RAG Memory Engine for Agentic AI Workbench

This module provides a **RAG-style (Retrieval-Augmented Generation) memory** for the agentic-ai-workbench, enabling cross-run learning and recall of prior pipeline outcomes.

## Overview

The memory engine captures:
- ✅ Compile errors and fixes
- ✅ Build errors and fixes  
- ✅ Auto-fix iterations
- ✅ Successful pipeline runs
- ✅ Model-specific patterns

And provides **keyword-based search** (FTS5 BM25 ranking) to retrieve relevant prior runs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE SOURCES                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐         │
│  │  Static KB          │    │  RAG Memory         │         │
│  │  (nntrainer_kb.py)  │    │  (memory_store.py)  │         │
│  │                     │    │                     │         │
│  │  • API reference    │    │  • Run history      │         │
│  │  • Layer catalog    │    │  • Error patterns   │         │
│  │  • Property types   │    │  • Fix patterns     │         │
│  │  • Forbidden symbols│    │  • Success patterns │         │
│  │                     │    │                     │         │
│  │  Source: Manual     │    │  Source: Auto-capture│        │
│  │  Scope: "How to"    │    │  Scope: "What happened"│      │
│  └─────────────────────┘    └─────────────────────┘         │
│              ↓                          ↓                    │
│              └──────────┬───────────────┘                    │
│                         ↓                                    │
│              ┌─────────────────────┐                         │
│              │  Recall API         │                         │
│              │  (recall.py)        │                         │
│              └─────────────────────┘                         │
│                         ↓                                    │
│              ┌─────────────────────┐                         │
│              │  LLM Code Gen       │                         │
│              │  (prompt context)   │                         │
│              └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `memory_store.py` | SQLite + FTS5 storage engine |
| `recall.py` | High-level recall API for agents |
| `nntrainer_kb.py` | Static API reference (unchanged) |
| `causallm_kb.py` | Layer catalog (unchanged) |
| `__init__.py` | Module exports |

## Storage

Memory is stored in:
```
nntrainer/workbench_output/run_memory.db
```

This is an SQLite database with FTS5 full-text search.

## Usage

### From Python Code

```python
from knowledge import (
    recall, 
    recall_errors, 
    recall_success_patterns,
    format_recall_context,
    get_memory_stats
)

# Search for similar errors
prior_errors = recall("compile error undefined reference", model="Qwen3", limit=3)

# Get error patterns for a model family
patterns = recall_errors("Qwen3", error_type="missing_include")

# Get success patterns
successes = recall_success_patterns("Llama3")

# Format as LLM prompt context
context = format_recall_context("compile error", model="Qwen3")
print(context)
# Output:
# ## Prior Run Insights for Qwen3
# 
# ### Run 1: Qwen3-1.8B (✅ Success)
# Error: undefined reference to nntrainer::Layer
# Fix: Added LINK_LIBRARIES(nntrainer)
# ...

# Get memory stats
stats = get_memory_stats("Qwen3")
print(f"Total runs: {stats['total_episodes']}")
print(f"Success rate: {stats['successful_runs'] / max(stats['total_episodes'], 1) * 100:.1f}%")
```

### In Agent Prompts

```python
from knowledge import recall, format_recall_context

def run(state: dict) -> dict:
    model_name = state["model_name"]
    
    # Get prior run insights
    context = format_recall_context("compile error", model=model_name, limit=3)
    
    # Inject into system prompt
    state["system_prompt"] += f"\n\n{context}"
    
    # Now LLM knows what errors to avoid
    ...
```

### Manual Query (for debugging)

```bash
# Open Python in the workbench directory
cd tools/agentic-ai-workbench/extension
python3 -c "
from knowledge import get_memory_stats, recall_errors
print('Stats:', get_memory_stats())
print('Qwen3 errors:', recall_errors('Qwen3'))
"
```

## API Reference

### `recall(query, model=None, event_type=None, limit=5)`

Search memory for similar prior runs.

**Args:**
- `query`: Search query (keywords)
- `model`: Model name (e.g., "Qwen3-1.8B") or family (e.g., "Qwen3")
- `event_type`: Filter by event type ("compile_error", "build_error", etc.)
- `limit`: Max results to return

**Returns:** List of matching episodes with metadata

### `recall_errors(model, error_type=None, limit=10)`

Get error patterns for a model family.

**Args:**
- `model`: Model name or family
- `error_type`: Optional filter (e.g., "undefined_reference")
- `limit`: Max results

**Returns:** List of error patterns with occurrence counts

### `recall_success_patterns(model, limit=5)`

Get successful run patterns for a model family.

**Args:**
- `model`: Model name or family
- `limit`: Max results

**Returns:** List of successful runs

### `format_recall_context(query, model=None, limit=3)`

Format recall results as context for LLM prompts.

**Args:**
- `query`: Search query
- `model`: Model name/family
- `limit`: Max results

**Returns:** Formatted string for injection into LLM prompt

### `get_memory_stats(model=None)`

Get memory store statistics.

**Args:**
- `model`: Optional model family filter

**Returns:** Dictionary with stats (total_episodes, successful_runs, failed_runs, etc.)

## Event Types

The following event types are captured:

| Event Type | Description |
|------------|-------------|
| `compile_error` | Compilation failed |
| `compile_success` | Compilation succeeded |
| `build_error` | CausalLM build failed |
| `build_success` | CausalLM build succeeded |
| `auto_fix_applied` | Auto-fix was applied |
| `pipeline_complete` | Overall pipeline finished |

## Error Types

Errors are classified into:

| Error Type | Pattern |
|------------|---------|
| `undefined_reference` | "undefined reference", "undefined symbol" |
| `missing_file` | "not found", "no such file" |
| `missing_include` | "missing include", "#include" |
| `undefined_symbol` | "undefined" |
| `type_error` | "type mismatch", "expected" |
| `segfault` | "segmentation fault", "segfault" |
| `timeout` | "timeout" |
| `permission_error` | "permission", "access denied" |
| `memory_error` | "memory", "out of memory" |
| `unknown` | Other errors |

## Clearing Memory

```python
from knowledge import get_memory_store

memory = get_memory_store()

# Clear all memory
memory.clear()

# Clear for specific model family only
memory.clear(model_family="Qwen3")
```

## Semantic Search (Phase 2+)

The memory engine now supports **hybrid search** combining FTS5 keyword matching with semantic similarity:

```python
from knowledge import semantic_recall, get_cross_model_learnings, get_model_insights

# Semantic search with embedding-based re-ranking
results = semantic_recall("undefined reference linker error", model="Qwen3", limit=5)

# Get similarity scores for each result
for r in results:
    print(f"Model: {r['model_name']}, Similarity: {r.get('semantic_similarity', 'N/A')}")
```

### Embedding Backends

The system auto-detects available embedding backends in this order:
1. **Ollama** (recommended, local) - `ollama pull nomic-embed-text`
2. **sentence-transformers** (local Python) - `pip install sentence-transformers`
3. **Mock** (fallback, deterministic pseudo-random)

To use a specific backend:
```python
from knowledge.embeddings import create_embedder

# Use Ollama explicitly
embedder = create_embedder(backend="ollama", model_name="nomic-embed-text")

# Use HuggingFace API
embedder = create_embedder(backend="huggingface", model_name="BAAI/bge-small-en-v1.5")

# Use sentence-transformers
embedder = create_embedder(backend="sentence-transformers", model_name="all-MiniLM-L6-v2")
```

## Cross-Model Learning (Phase 2+)

Transfer knowledge across model families:

```python
from knowledge import get_cross_model_learnings, get_model_insights

# Find patterns from Qwen2 that apply to Qwen3
learnings = get_cross_model_learnings("Qwen2", "Qwen3", limit=5)
for l in learnings:
    print(f"Transferable: {l['error_type']}")
    print(f"  Fix: {l['fix_applied']}")
    print(f"  Reason: {l['reason']}")

# Get comprehensive model insights
insights = get_model_insights("Qwen3")
print(f"Stats: {insights['stats']}")
print(f"Common errors: {insights['common_errors']}")
print(f"Cross-model learnings from related families: {insights['cross_model_learnings']}")
```

### How Cross-Model Learning Works

1. **Related Family Detection**: Identifies model families with shared architecture roots (e.g., Qwen2 → Qwen3, Llama2 → Llama3)
2. **Pattern Matching**: Finds error types that appear in both source and target families
3. **Transfer Scoring**: Assigns confidence based on error type similarity and fix availability
4. **Insight Aggregation**: Combines stats, errors, successes, and cross-model patterns

## UI Integration (Phase 3)

The webview now includes a **Run History** panel:
- Click "📚 Run History" button in the topbar
- Search by keyword with optional model family filter
- View error messages, fixes applied, and timestamps
- Results ranked by relevance (FTS5 BM25)

## Troubleshooting

### Memory store not found
The database is created on first run. If it doesn't exist, run a pipeline and it will be auto-created.

### Search returns no results
This is expected for a fresh installation. Run some pipelines first to populate the memory.

### Slow search performance
FTS5 is very fast for keyword search. If you experience slowness, check:
- Database file size (should be < 100 MB for typical use)
- Number of episodes (consider `clear()` for old data)

## Comparison with Static KB

| Aspect | Static KB (`nntrainer_kb.py`) | RAG Memory (`memory_store.py`) |
|--------|------------------------------|-------------------------------|
| Content | API reference, layer catalog | Run history, error patterns |
| Source | Manually curated | Auto-captured from runs |
| Updates | Manual (when API changes) | Automatic (each pipeline run) |
| Search | Keyword only | FTS5 BM25 + recency ranking |
| Scope | "How to use nntrainer API" | "What happened in prior runs" |
| Lifetime | Permanent (ships with code) | Grows over time |

Both work together: Static KB provides API correctness, RAG Memory provides experiential learning.
