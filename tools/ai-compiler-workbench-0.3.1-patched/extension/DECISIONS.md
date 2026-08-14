# Architecture Decisions Log

This document records all significant architectural decisions made in the AI Compiler Workbench extension, including the context, options considered, and rationale for each decision.

---

## Decision: Two-Graph View Design

**Date:** 2024  
**Status:** Accepted  
**Context:** Need to visualize both source model architecture and target nntrainer implementation.

### Options Considered

1. **Single combined graph** - Show both source and target nodes in one graph
2. **Three tabs** - Model Graph, Mapping, nntrainer Graph
3. **Two side-by-side graphs with click-to-highlight** (Selected)

### Decision

Use two side-by-side graph panes with click-to-highlight mapping between corresponding nodes.

### Rationale

- **Cognitive load:** Two focused views are easier to understand than one complex combined graph
- **Mapping clarity:** Click-to-highlight provides immediate visual feedback on correspondences
- **Screen real estate:** Side-by-side allows direct comparison without tab switching
- **Simplicity:** Eliminates need for dedicated "Mapping" tab

### Consequences

- Requires maintaining node mapping table
- Cross-highlighting logic in webview
- Both graphs must use compatible layout algorithms

---

## Decision: Weight Preview Cache

**Date:** 2024  
**Status:** Accepted  
**Context:** Need to show weight tensor statistics when user clicks graph nodes.

### Options Considered

1. **Read weights on-demand** - Open safetensors file each time node is clicked
2. **Pre-load all weights into state** - Include full weight data in graph view
3. **Build preview cache once per pipeline** (Selected)

### Decision

Build a weight preview cache at nntrainer_lowering time, store in module-level variable.

### Rationale

- **Performance:** Reading safetensors once is faster than opening file per click
- **Memory:** Cache only contains statistics (min/max/mean/std/5 values), not full tensors
- **Clean separation:** Cache is built when weights are available, used by graph views
- **Cleanup:** Cache cleared at start of each new pipeline

### Consequences

- Requires `clear_weight_cache()` call at pipeline start
- Cache is process-global (acceptable for single-pipeline-per-process design)
- Weight preview only available after nntrainer_lowering runs

---

## Decision: Background Weight Download

**Date:** 2024  
**Status:** Accepted  
**Context:** Weight downloads can take minutes; graph generation should not wait.

### Options Considered

1. **Sequential** - Wait for weights before any graph processing
2. **Full parallel** - Run weight download completely independent
3. **Background thread with join** (Selected)

### Decision

Start weight download in background thread, continue with graph/.ini/.cpp generation, join before artifact collection.

### Rationale

- **User experience:** Graph visualization available quickly, regardless of weight size
- **Correctness:** Weights only needed for final artifact listing, not graph generation
- **Resource efficiency:** Network I/O happens in parallel with CPU-bound graph processing
- **Error handling:** Pipeline can succeed even if weight download fails

### Consequences

- Requires thread-safe result storage (`_bg_results`, `_bg_lock`)
- Must handle case where weights never arrive (timeout/failure)
- Artifact manager waits for weights before listing

---

## Decision: LangGraph StateGraph

**Date:** 2024  
**Status:** Accepted  
**Context:** Need to coordinate multiple agents with conditional retry logic.

### Options Considered

1. **Manual orchestration** - Call agents sequentially in orchestrator
2. **LangGraph StateGraph** (Selected)
3. **Custom graph executor**

### Decision

Use LangGraph StateGraph with nodes for each agent and conditional edges for retry logic.

### Rationale

- **Retry handling:** Conditional edges naturally express Compiler -> Auto-Fix -> Compiler loop
- **State management:** PipelineState flows through all nodes automatically
- **Extensibility:** Easy to add new agents or modify flow
- **Fallback:** Sequential execution available if langgraph not installed

### Consequences

- Additional dependency (langgraph)
- State must be JSON-serializable for some use cases
- Recursion limit must be configured for retry loops

---

## Decision: Semantic IR Before Lowering

**Date:** 2024  
**Status:** Accepted  
**Context:** Need architecture-agnostic model representation before nntrainer-specific conversion.

### Options Considered

1. **Direct to nntrainer** - Parse HF config directly to nntrainer graph
2. **Two-stage** - Build semantic IR first, then lower to nntrainer (Selected)

### Decision

Build architecture-level Semantic IR (CausalLMIR) first, then lower to nntrainer graph.

### Rationale

- **Separation of concerns:** Architecture understanding separate from target codegen
- **Validation:** Can validate semantic IR before lowering
- **Reusability:** Same semantic IR could lower to different targets
- **Diagnostics:** Clear error messages about architecture vs lowering issues

### Consequences

- Two data structures to maintain (CausalLMIR, nntrainer graph)
- Adapter pattern required for each architecture
- Additional processing step in pipeline

---

## Decision: Uniform Layer Detection

**Date:** 2024  
**Status:** Accepted  
**Context:** Generated C++ for 28-layer model would be huge if each layer fully unrolled.

### Options Considered

1. **Always unroll** - Generate code for every layer explicitly
2. **Always loop** - Generate single reusable function, call in loop
3. **Detect uniform layers** (Selected)

### Decision

Detect if all decoder layers share same structural signature; generate reusable functions if uniform, unroll if not.

### Rationale

- **Code size:** 28-layer model with unrolled code = thousands of lines
- **Correctness:** Some models may have non-uniform layers (rare but possible)
- **Performance:** Same generated code either way (compile-time difference only)
- **Validation:** `uniform_layer_signature()` computed once, checked for all layers

### Consequences

- Requires structural signature comparison
- C++ generator has two modes (unrolled vs loop)
- Must verify all layers match before using uniform mode

---

## Decision: Weight Cache Cleanup on Pipeline Start

**Date:** 2024  
**Status:** Accepted  
**Context:** Old weight data from previous pipeline run should not persist.

### Options Considered

1. **No cleanup** - Let cache persist across runs
2. **Cleanup at end** - Clear cache after pipeline completes
3. **Cleanup at start** (Selected)

### Decision

Clear weight cache at the very beginning of `run_pipeline()`.

### Rationale

- **Correctness:** Ensures no stale data from previous model
- **Simplicity:** One clear point of cleanup, not scattered
- **Memory:** Frees memory early for new pipeline
- **Debugging:** Easier to reason about cache state

### Consequences

- Must import `graph_views` in `orchestrator.py`
- Cache always empty at pipeline start
- New cache built during nntrainer_lowering

---

## Decision: Inspector Tabs (Properties/Weights)

**Date:** 2024  
**Status:** Accepted  
**Context:** Node inspector needs to show both structural properties and weight data.

### Options Considered

1. **Single panel** - Show all info in one long panel
2. **Two tabs** - Properties tab, Weights tab (Selected)
3. **Expandable sections** - Collapsible sections within panel

### Decision

Use two tabs in inspector: Properties and Weights.

### Rationale

- **Organization:** Clear separation between structural and weight data
- **Screen space:** Tabs keep panel compact
- **Consistency:** Matches VSCode property panel patterns
- **Optional data:** Weights tab shows "No weight data" for non-weight nodes

### Consequences

- Tab state management in webview (`inspTab`)
- Two different render paths in `renderInspector()`
- Weight info must be present in node data

---

## Decision: Graph Layout Algorithm

**Date:** 2024  
**Status:** Accepted  
**Context:** Need readable layout for graphs with 30+ nodes.

### Options Considered

1. **Simple topological** - Just assign y by depth, x by order
2. **Sugiyama with barycenter** (Selected)
3. **Force-directed** - Physics-based layout

### Decision

Use Sugiyama-style layered layout with barycenter crossing reduction.

### Rationale

- **Layered structure:** Neural networks naturally layer
- **Crossing reduction:** Barycenter sorting untangles edges
- **Deterministic:** Same graph always produces same layout
- **Performance:** Fast enough for 50-100 node graphs

### Consequences

- `_layout_vertical()` function with 3 stages
- `BARYCENTER_SWEEPS = 4` for good results
- Row centering for visual balance

---

## Decision: File Structure

**Date:** 2024  
**Status:** Accepted  
**Context:** Need logical organization of extension code.

### Structure

```
extension/
├── engine/
│   ├── agents/          # All agent implementations
│   ├── api/             # Core APIs (semantic model, adapters, lowering)
│   ├── converters/      # Code/data format converters
│   ├── utils/           # Utility functions (weight_reader.py)
│   └── core/            # Core infrastructure (events, state)
├── webview/
│   └── main.html        # Single-file webview UI
├── EXECUTION_FLOW.md    # This document
└── DECISIONS.md         # Architecture decisions
```

### Rationale

- **Agents separate from API:** Agents orchestrate, API provides building blocks
- **Single webview file:** Easier to reason about UI as one unit
- **Utils for shared code:** weight_reader used by multiple agents
- **Documentation in repo:** EXECUTION_FLOW and DECISIONS alongside code

---

## Decision: No LLM in Core Pipeline

**Date:** 2024  
**Status:** Accepted  
**Context:** Many agents could potentially use LLM for suggestions.

### Decision

Core pipeline (model_discovery through artifact_manager) uses no LLM calls. Only auto_fix agent uses LLM for fix suggestions.

### Rationale

- **Determinism:** Same input always produces same output
- **Speed:** No network latency for core functionality
- **Cost:** No API costs for basic operation
- **Debugging:** Easier to trace and reproduce issues

### Consequences

- Compatibility suggestions are rule-based
- Auto-fix requires separate API key
- Core pipeline works offline (after initial config download)

---

## Decision: Weight Preview Statistics

**Date:** 2024  
**Status:** Accepted  
**Context:** What statistics to show for weight preview?

### Decision

Show: min, max, mean, std, first 5 values

### Rationale

- **Min/Max:** Range of values (detect NaN/Inf issues)
- **Mean:** Center of distribution (should be ~0 for initialized weights)
- **Std:** Spread of values (detect collapsed or exploded weights)
- **First 5:** Concrete examples (verify dtype conversion worked)

### Consequences

- Requires numpy operations on tensor data
- Statistics computed once during cache build
- Webview displays with 6 decimal precision

---

## Decision: Event Bus Pattern

**Date:** 2024  
**Status:** Accepted  
**Context:** Need clean communication between backend and webview.

### Decision

Use centralized event bus (`bus` object) for all backend-to-webview communication.

### Rationale

- **Decoupling:** Agents don't need to know about webview implementation
- **Consistency:** All events go through same channel
- **Testing:** Easy to mock bus for unit tests
- **Extensibility:** New event types added in one place

### Consequences

- `from .events import bus` in every agent
- Event types defined in core/events.py
- Extension forwards bus events to webview via postMessage

---

## Decision: State Persistence

**Date:** 2024  
**Status:** Accepted  
**Context:** Pipeline state should persist across VSCode restarts.

### Decision

Save `state.json` at end of each pipeline run in output directory.

### Rationale

- **Debugging:** Can inspect state after run completes
- **Resume:** Future feature could resume from saved state
- **Audit:** Record of what was generated and when

### Consequences

- Must sanitize state (remove API keys)
- JSON serialization for all state values
- State file in user-visible output directory

---

## Decision: Cross-Highlighting Implementation

**Date:** 2024  
**Status:** Accepted  
**Context:** Clicking node in one graph should highlight corresponding nodes in other graph.

### Decision

Store `nodeMappings` array in webview, use `crossHighlightFrom()` to find and highlight matches.

### Rationale

- **Efficiency:** Single lookup per click
- **Flexibility:** Supports one-to-many and many-to-one mappings
- **Clarity:** Mappings computed once at backend, used many times in frontend

### Consequences

- `build_node_mappings()` in graph_views.py
- `node_mappings` event sends mappings to webview
- Webview stores mappings in module-level variable

---

## Decision: Weight Reader Location

**Date:** 2024  
**Status:** Accepted  
**Context:** Where to place weight reading utility?

### Decision

Place in `engine/utils/weight_reader.py` as standalone module.

### Rationale

- **Reusability:** Multiple agents may need weight reading
- **Separation:** Not tied to any specific agent
- **Testing:** Easy to unit test independently

### Consequences

- Import as `from ..utils.weight_reader import build_weight_cache`
- No agent-specific logic in weight_reader
- Pure utility functions only
