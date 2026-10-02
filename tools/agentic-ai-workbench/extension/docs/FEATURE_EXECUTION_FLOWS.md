# Agentic AI Workbench - Feature Execution Flows

This document provides detailed execution flows for each feature in the AI Compiler Workbench extension. Each feature is documented separately with clear function call sequences and data flow.

---

## Table of Contents

1. [Pipeline Orchestration](#1-pipeline-orchestration)
2. [Model Discovery](#2-model-discovery)
3. [Compatibility Analysis](#3-compatibility-analysis)
4. [Weight Download & Conversion](#4-weight-download--conversion)
5. [NNTrainer Lowering](#5-nntrainer-lowering)
6. [INI File Generation](#6-ini-file-generation)
7. [Graph Building & Visualization](#7-graph-building--visualization)
8. [Dual Graph View](#8-dual-graph-view)
9. [C++ Code Generation](#9-c-code-generation)
10. [CausalLM Component Installation](#10-causallm-component-installation)
11. [Compilation](#11-compilation)
12. [Auto-Fix](#12-auto-fix)
13. [Profiling](#13-profiling)
14. [Artifact Management](#14-artifact-management)
15. [Chat Agent](#15-chat-agent)

---

## 1. Pipeline Orchestration

**File:** `engine/agents/orchestrator.py`

### Overview
The Orchestrator coordinates all agents in a LangGraph StateGraph, managing the pipeline flow from model discovery to artifact collection.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline()                                        │
│  1. graph_views.clear_weight_cache()                                         │
│  2. Create new PipelineState                                                 │
│  3. Build LangGraph StateGraph                                               │
│  4. Invoke graph with recursion_limit=60                                     │
│  5. Save state.json                                                          │
│  6. Send pipeline_complete event                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph Flow                                 │
│                                                                              │
│  model_discovery ──► start_weight_download (background thread)               │
│                          │                                                   │
│                          ▼                                                   │
│                    compatibility ──► nntrainer_lowering                      │
│                          │                                                   │
│                          ▼                                                   │
│                    ini_generator ──► graph_builder                           │
│                          │                                                   │
│                          ▼                                                   │
│                    cpp_generator ──► causallm_install                        │
│                          │                                                   │
│                          ▼                                                   │
│                    causallm_build_run ──► dual_graph                         │
│                          │                                                   │
│                          ▼                                                   │
│                    compiler ──┬──► auto_fix ──► dual_graph (retry loop)      │
│                               │           (max 2 iterations)                 │
│                               ▼                                              │
│                    profiler ──► join_weight_download                          │
│                          │                                                   │
│                          ▼                                                   │
│                    artifact_manager ──► END                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run_pipeline(model_name, out_dir, api_key, ...)`
   - `graph_views.clear_weight_cache()`
   - `new_state(...)` → Creates PipelineState dict
   - `build_graph()` → Constructs LangGraph StateGraph
   - `graph.invoke(state, config={"recursion_limit": 60})`

2. **Conditional Edge Logic** (`_should_retry_compile()`):
   - If `compiled=true` → goto `profiler`
   - If `nntrainer_missing` or "not found" in log → goto `profiler`
   - If `fix_iterations >= 2` → goto `profiler`
   - If no `anthropic_api_key` → goto `profiler`
   - Otherwise → goto `auto_fix`

### State Transitions

| Key | Set By | Description |
|-----|--------|-------------|
| `model_name` | User | Model identifier |
| `out_dir` | User | Output directory |
| `hf_config` | model_discovery | HuggingFace config |
| `architecture` | model_discovery | Model architecture name |
| `semantic_ir` | compatibility | Semantic IR dict |
| `nntrainer_graph_ir` | nntrainer_lowering | Lowered graph |
| `cpp_code` | cpp_generator | Generated C++ code |
| `compiled` | compiler | Compilation status |
| `profile` | profiler | Profiling results |
| `artifacts` | artifact_manager | Generated files list |

---

## 2. Model Discovery

**File:** `engine/agents/model_discovery.py`

### Overview
Discovers and extracts model metadata/architecture from HuggingFace Hub without downloading weights.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. bus.agent_status("model_discovery", "running")                           │
│  2. Import AutoConfig from transformers                                      │
│  3. config = AutoConfig.from_pretrained(model_name)                          │
│  4. hf_config = config.to_dict()                                             │
│  5. state["hf_config"] = hf_config                                           │
│  6. state["architecture"] = architectures[0] or type(config).__name__        │
│  7. Log discovered metadata                                                  │
│  8. bus.agent_status("model_discovery", "done", architecture)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `bus.agent_status("model_discovery", "running")`
   - `AutoConfig.from_pretrained(model_name)`
   - `config.to_dict()`
   - `bus.log(...)` - Logs hidden_size, num_layers, vocab_size
   - `bus.agent_status("model_discovery", "done", architecture)`

### Output State

```python
state = {
    "hf_config": { ... },        # Full config.json content
    "architecture": "LlamaForCausalLM",  # Primary architecture class
    "errors": []                 # Empty on success
}
```

---

## 3. Compatibility Analysis

**File:** `engine/agents/compatibility.py`

### Overview
Builds model architecture graph WITHOUT downloading weights, checks ops against nntrainer op table.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check cache freshness (30-day expiry)                                    │
│     │                                                                        │
│     ├─► Cache FRESH ──► Load cached graph_ir, report, semantic_ir           │
│     │                 ──► Return early                                       │
│     │                                                                        │
│     └─► Cache STALE ──► Continue below                                       │
│                                                                              │
│  2. Fetch config (AutoConfig.from_pretrained)                                │
│  3. Build module tree: model = AutoModel.from_config(config)                 │
│     (NO weight download - uses random tensors)                               │
│                                                                              │
│  4. Select architecture adapter via select_adapter(config, model)            │
│     │                                                                        │
│     ├─► Adapter MATCHED ──► semantic_ir = adapter.build_semantic_ir()       │
│     │                       ──► state["semantic_ir"] = semantic_ir.to_dict()│
│     │                       ──► state["semantic_capabilities"] = describe() │
│     │                                                                        │
│     └─► Adapter NOT FOUND ──► state["semantic_ir"] = None                   │
│                               ──► Fall back to module-tree tracing          │
│                                                                              │
│  5. Trace model: graph = GenericFxParser(model).parse()                      │
│     (Module-tree walker, NOT torch.fx)                                       │
│                                                                              │
│  6. Analyze: report = OpLevelCompatibilityChecker().analyze(graph)          │
│  7. Export: graph_ir = graph.export()                                        │
│  8. Cache results                                                            │
│  9. If unsupported ops exist:                                                │
│     └─► _maybe_suggest(state, unsupported) ──► LLM suggestions (optional)   │
│                                                                              │
│  10. bus.agent_status("compatibility", "done", "...% compatible")            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `cache.cache_root_for(out_dir, model_name)`
   - `cache.is_fresh(meta, "graph")` - Checks 30-day cache
   - `_load_cached_graph(cache_root)` - Loads cache if fresh
   - `AutoConfig.from_pretrained(model_name)`
   - `AutoModel.from_config(config)` - No weight download
   - `select_adapter(config, model)` - Finds architecture adapter
   - `adapter.build_semantic_ir(config, model)` - Builds semantic IR
   - `describe_semantic_capabilities(semantic_ir)`
   - `GenericFxParser(model, model_name).parse()`
   - `OpLevelCompatibilityChecker().analyze(graph)`
   - `graph.export()`
   - `_save_cached_graph(cache_root, ...)`
   - `_maybe_suggest(state, unsupported)` - Optional LLM suggestions

### Cache Schema (v2)

```json
{
  "graph_ir.json": { ... },
  "report.json": {
    "summary": {
      "compatibility": 85.7,
      "supported_nodes": 30,
      "unsupported_nodes": 5
    },
    "unsupported": [...]
  },
  "semantic_ir.json": {
    "semantic_ir": { ... },
    "semantic_capabilities": { ... }
  }
}
```

---

## 4. Weight Download & Conversion

**Files:** `engine/agents/weight_download.py`, `engine/agents/weight_converter.py`

### Overview
Downloads model weights from HuggingFace Hub and converts to nntrainer binary format. Runs in background thread.

### Weight Download Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    start_weight_download(state)                              │
│                                                                              │
│  1. Create background thread: _weight_worker()                               │
│  2. Thread.start() - Runs concurrently                                       │
│  3. Return immediately (non-blocking)                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    _weight_worker(run_id, model_name, out_dir, api_key)     │
│                                                                              │
│  1. local_state = { model_name, out_dir, anthropic_api_key }                │
│  2. weight_download.run(local_state)                                         │
│  3. weight_converter.run(local_state)                                        │
│  4. Store results in _bg_results[run_id] (thread-safe)                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    join_weight_download(state)                               │
│                                                                              │
│  1. Wait for background thread to complete                                   │
│  2. Merge results: state["weights_path"], state["converted_weights_path"]   │
│  3. Merge any errors                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Weight Download Function Sequence

1. `run(state: dict) -> dict`
   - `cache.cache_root_for(out_dir, model_name)`
   - `cache.is_fresh(meta, "weights")` - Checks 30-day cache
   - `snapshot_download(repo_id=model_name, ...)` - Downloads weights
   - `cache.set_entry(cache_root, "weights", {...})`
   - `_dir_size_mb(path)` - Calculates size

### Weight Conversion Function Sequence

1. `run(state: dict) -> dict`
   - `cache.is_fresh(meta, "converted")` - Checks cache
   - `Path(weights_path).glob("*.safetensors")` - Finds files
   - `safe_open(st_file, framework="pt"|"np")` - Opens tensors
   - For each tensor:
     - `f.get_tensor(key)` - Gets tensor
     - Convert dtype (bfloat16 → float32 if needed)
     - `tensor.numpy().tobytes()` or manual conversion
     - Write to `converted_weights.bin`
   - `cache.set_entry(cache_root, "converted", {...})`

### Output State

```python
state = {
    "weights_path": "/path/to/weights",           # safetensors directory
    "converted_weights_path": "/path/to/converted_weights.bin"  # Binary file
}
```

---

## 5. NNTrainer Lowering

**File:** `engine/agents/nntrainer_lowering.py`

### Overview
Converts semantic IR to nntrainer-specific graph, builds weight preview cache.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check state["semantic_ir"] exists                                        │
│     └─► If None ──► Log "skipping nntrainer lowering" ──► Return            │
│                                                                              │
│  2. model_ir = CausalLMIR.from_dict(semantic_ir_dict)                        │
│                                                                              │
│  3. If weights available:                                                    │
│     └─► build_weight_cache(weights_path)                                     │
│         └─► graph_views.set_weight_cache(cache)                              │
│         └─► Reads all tensors, extracts: preview, min, max, mean, std        │
│                                                                              │
│  4. ThreadPoolExecutor (2 workers):                                          │
│     │                                                                        │
│     ├─► Worker 1: graph_views.build_model_graph_view(model_ir)              │
│     │   └─► Builds semantic-level graph for webview                         │
│     │                                                                        │
│     └─► Worker 2: _lower_and_validate(model_ir)                              │
│         ├─► NNTrainerLowerer(model_ir).lower()                              │
│         └─► validate(graph, model_ir)                                        │
│                                                                              │
│  5. state["model_graph_view"] = model_view_future.result()                  │
│  6. nntrainer_graph, diagnostics = target_future.result()                    │
│  7. nntrainer_graph_ir = nntrainer_graph.export()                            │
│  8. state["nntrainer_graph_ir"] = nntrainer_graph_ir                         │
│  9. state["lowering_diagnostics"] = diagnostics.to_dict()                    │
│  10. state["node_mappings"] = graph_views.build_node_mappings(...)          │
│  11. state["nntrainer_graph_view"] = graph_views.build_nntrainer_graph_view()│
│  12. Log validation errors if any                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `CausalLMIR.from_dict(semantic_ir_dict)`
   - `build_weight_cache(weights_path)` (if weights available)
   - `graph_views.set_weight_cache(weight_cache)`
   - `ThreadPoolExecutor(max_workers=2)`:
     - `graph_views.build_model_graph_view(model_ir)`
     - `NNTrainerLowerer(model_ir).lower()`
     - `validate(graph, model_ir)`
   - `graph_views.build_node_mappings(model_ir, nntrainer_graph)`
   - `graph_views.build_nntrainer_graph_view(nntrainer_graph_ir)`

### Weight Preview Cache Structure

```python
weight_cache = {
    "model.layers.0.self_attn.q_proj.weight": {
        "preview": [0.012, -0.034, 0.056, ...],  # First 5 values
        "min": -0.523,
        "max": 0.612,
        "mean": 0.001,
        "std": 0.089
    },
    ...
}
```

---

## 6. INI File Generation

**File:** `engine/agents/ini_generator.py`

### Overview
Generates nntrainer's `model.ini` from the traced IR graph with proper input_layers resolution.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check emission_mode:                                                     │
│     └─► If "causallm_component" ──► Skip (INI not applicable) ──► Return   │
│                                                                              │
│  2. graph_ir = state["nntrainer_graph_ir"] or state["graph_ir"]             │
│                                                                              │
│  3. Build id_to_section mapping:                                             │
│     └─► Only for supported nodes with node_type                              │
│                                                                              │
│  4. Define resolve_inputs(node):                                             │
│     └─► Walk back through passthrough nodes                                  │
│     └─► Find nearest supported ancestor(s)                                   │
│                                                                              │
│  5. Generate INI content:                                                    │
│     ├─► [Model] section                                                      │
│     └─► For each supported node:                                             │
│         ├─► [{section_name}]                                                 │
│         ├─► Type = {node_type}                                               │
│         ├─► {attributes}                                                     │
│         ├─► ; weight: name=... shape=... dtype=... params=...               │
│         └─► input_layers = {resolved_inputs}                                 │
│                                                                              │
│  6. Write to {out_dir}/generated/model.ini                                   │
│  7. state["ini_content"] = content                                           │
│  8. state["ini_path"] = ini_path                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `_section_name(name)` - Converts names (replaces `.`, `/`, `-` with `_`)
   - `resolve_inputs(node, seen)` - Recursive input resolution
   - Writes `model.ini` to disk
   - `bus.log(f"Generated model.ini ...")`

### INI File Format

```ini
[Model]
Model = LlamaForCausalLM
Type = NeuralNetwork
Epochs = 1
Loss = cross
Save_Path = model.bin

[embed_tokens]
Type = embedding
input_layers =

[layers_0_input_layernorm]
Type = layer_norm
input_layers = embed_tokens
; weight: name=model.layers.0.input_layernorm.weight shape=[4096] dtype=float32 params=4096

[layers_0_self_attn_q_proj]
Type = fully_connected
input_layers = layers_0_input_layernorm
; weight: name=model.layers.0.self_attn.q_proj.weight shape=[4096,4096] dtype=float32 params=16777216
```

---

## 7. Graph Building & Visualization

**File:** `engine/agents/graph_builder.py`, `engine/agents/graph_views.py`

### Overview
Builds internal graph view with layout coordinates for webview visualization.

### Graph Builder Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    graph_builder.run(state)                                  │
│                                                                              │
│  1. graph_ir = state["graph_ir"] or state["nntrainer_graph_ir"]             │
│                                                                              │
│  2. Build view_nodes:                                                        │
│     └─► For each node: id, label, type, status, attributes, shapes          │
│                                                                              │
│  3. Build view_edges:                                                        │
│     └─► {id: "src-tgt", source: src_id, target: tgt_id}                      │
│                                                                              │
│  4. _layout_vertical(view_nodes, view_edges, order)                          │
│     └─► Assigns x/y coordinates to each node                                 │
│                                                                              │
│  5. Write graph.json to disk                                                 │
│  6. state["graph_view"] = {nodes, edges}                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layout Algorithm (`_layout_vertical`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _layout_vertical(nodes, edges, order)                     │
│                                                                              │
│  STAGE 1: Depth Assignment                                                   │
│  ├─► Initialize depth[nid] = 0 for all nodes                                │
│  └─► Iterate: depth[tgt] = max(depth[src] + 1) for all edges               │
│                                                                              │
│  STAGE 2: Row Assignment                                                     │
│  ├─► Group nodes by depth: rows[depth] = [node_ids]                         │
│  └─► Sort each row by declaration order                                     │
│                                                                              │
│  STAGE 3: Barycenter Crossing Reduction (4 sweeps)                          │
│  ├─► For each sweep:                                                        │
│  │   ├─► Top-to-bottom: Sort by parent barycenter                          │
│  │   └─► Bottom-to-top: Sort by child barycenter                           │
│  └─► Reindex positions after each sort                                      │
│                                                                              │
│  STAGE 4: Position Assignment                                                │
│  ├─► Calculate row offsets for centering                                     │
│  └─► Assign: node.x = (offset + i) * COL_WIDTH                              │
│              node.y = depth * ROW_HEIGHT                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Graph View (Semantic IR)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    build_model_graph_view(model_ir)                          │
│                                                                              │
│  1. Create embedding node                                                    │
│                                                                              │
│  2. For each decoder layer:                                                  │
│     ├─► If n_layers <= 3 OR first/last layer:                               │
│     │   └─► _build_expanded_layer() - Full expansion                        │
│     │       ├─► input_norm                                                  │
│     │       ├─► q_proj, k_proj, v_proj                                      │
│     │       ├─► q_norm, k_norm (if present)                                 │
│     │       ├─► mha_core                                                    │
│     │       ├─► o_proj                                                      │
│     │       ├─► residual_add                                                │
│     │       ├─► post_attention_norm                                         │
│     │       ├─► gate_proj, up_proj (if gated MLP)                           │
│     │       ├─► activation                                                  │
│     │       ├─► down_proj                                                   │
│     │       └─► mlp_residual                                                │
│     │                                                                        │
│     └─► Else:                                                               │
│         └─► Collapsed node: "Decoder Layer {i}"                             │
│                                                                              │
│  3. Create final_norm node                                                   │
│  4. Create lm_head node                                                      │
│  5. Connect all edges                                                        │
│  6. _layout_vertical(nodes, edges, order)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### NNTrainer Graph View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    build_nntrainer_graph_view(nntrainer_graph_ir)           │
│                                                                              │
│  1. Group nodes by group_id (decoder layers)                                 │
│                                                                              │
│  2. Determine collapse set:                                                  │
│     └─► Middle layers if n_layers > 3                                       │
│                                                                              │
│  3. Build view_nodes:                                                        │
│     ├─► Non-collapsed nodes: Keep as-is                                     │
│     │   └─► Add weightInfo from cache                                       │
│     │                                                                        │
│     └─► Collapsed groups: Create single collapsed node                       │
│         └─► Aggregate source_node_ids                                        │
│                                                                              │
│  4. Redirect edges through collapsed nodes                                   │
│  5. _layout_vertical(view_nodes, view_edges, order)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Node Mappings (Cross-Highlighting)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    build_node_mappings(model_ir, nntrainer_graph)           │
│                                                                              │
│  1. Group nntrainer nodes by group_id                                        │
│                                                                              │
│  2. For each decoder layer group:                                            │
│     └─► Mapping: {                                                           │
│         "sourceIds": [semantic_node_ids],                                    │
│         "targetIds": [nntrainer_node_ids],                                   │
│         "mappingType": "many_to_one"|"one_to_one",                          │
│         "description": "Decoder layer {i}"                                   │
│       }                                                                      │
│                                                                              │
│  3. For ungrouped nodes:                                                     │
│     └─► One-to-one mappings                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Dual Graph View

**File:** `engine/agents/dual_graph.py`

### Overview
Publishes two graphs to webview with cross-highlighting support.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check state["nntrainer_graph_ir"]:                                       │
│     │                                                                        │
│     ├─► EXISTS (semantic adapter matched)                                    │
│     │   └─► _publish_semantic_graphs(state)                                  │
│     │       ├─► bus.graph(model_graph_view, target="model")                 │
│     │       ├─► bus.graph(nntrainer_graph_view, target="nntrainer")         │
│     │       ├─► bus.file_content("nntrainer", "model.ini", ini_content)     │
│     │       └─► bus.file_content("nntrainer", "generated_model.cpp", code)  │
│     │                                                                        │
│     └─► NONE (fallback mode)                                                 │
│         └─► _publish_fallback_graphs(state)                                  │
│             ├─► build_ini_graph(ini_content) ──► Model Graph               │
│             └─► build_cpp_graph(cpp_code) ──► nntrainer Graph              │
│                                                                              │
│  2. bus.node_mappings(state["node_mappings"])                                │
│  3. Enable click-to-highlight in webview                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `_publish_semantic_graphs(state)` or `_publish_fallback_graphs(state)`
   - `build_ini_graph(ini_content)` (fallback only)
   - `build_cpp_graph(cpp_code)` (fallback only)
   - `bus.node_mappings(mappings)`

### Graph View Data Structure

```json
{
  "nodes": [
    {
      "id": "layers_0_self_attn_q_proj",
      "label": "layers_0_self_attn_q_proj",
      "type": "fully_connected",
      "status": "mapped",
      "x": 200,
      "y": 390,
      "weightInfo": {
        "name": "model.layers.0.self_attn.q_proj.weight",
        "shape": [4096, 4096],
        "dtype": "float32",
        "params": 16777216,
        "preview": [0.012, -0.034, ...],
        "min": -0.523,
        "max": 0.612,
        "mean": 0.001,
        "std": 0.089
      }
    }
  ],
  "edges": [
    {"source": "layers_0_input_layernorm", "target": "layers_0_self_attn_q_proj"}
  ]
}
```

---

## 9. C++ Code Generation

**File:** `engine/agents/cpp_generator_agent.py`

### Overview
Generates C++ code from graph IR in two modes: MODEL_API or CAUSALLM_COMPONENT.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. nntrainer_graph_ir = state.get("nntrainer_graph_ir")                    │
│  2. graph_ir = nntrainer_graph_ir or state["graph_ir"]                      │
│  3. graph = _graph_from_ir(graph_ir)                                         │
│  4. is_causallm_component = graph.metadata.emission_mode == "causallm_component" │
│                                                                              │
│  5. Branch based on emission mode:                                           │
│     │                                                                        │
│     ├─► MODEL_API mode:                                                      │
│     │   └─► _run_model_api(state, graph, generated_dir)                      │
│     │       ├─► code = CPPGenerator(graph).generate()                        │
│     │       ├─► Inject LLM suggestions (if any)                              │
│     │       ├─► Append main() for smoke test                                 │
│     │       ├─► Write generated_model.cpp                                    │
│     │       └─► bus.code("generated_model.cpp", code)                        │
│     │                                                                        │
│     └─► CAUSALLM_COMPONENT mode:                                             │
│         └─► _run_causallm_component(state, graph, generated_dir)             │
│             ├─► files = CPPGenerator(graph).generate_component()             │
│             ├─► Write generated_<arch>_causallm.h                            │
│             ├─► Write generated_<arch>_causallm.cpp                          │
│             ├─► _write_manifests(state, causallm_dir)                        │
│             │   ├─► build_model_metadata(model_ir) → config.json            │
│             │   └─► build_weight_manifest(model_ir) → weight_manifest.json  │
│             └─► bus.code(...) for both files                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `_graph_from_ir(graph_ir)` - Reconstructs Graph object from IR
   - `CPPGenerator(graph).generate()` (MODEL_API)
   - `CPPGenerator(graph).generate_component()` (CAUSALLM_COMPONENT)
   - `_inject_suggestions(code, suggestions)` (MODEL_API only)
   - `_write_manifests(state, causallm_dir)` (CAUSALLM_COMPONENT only)

### MODEL_API Output

```cpp
#include <nntrainer/nntrainer.h>

std::unique_ptr<nntrainer::Model> buildModel() {
    auto model = std::make_unique<nntrainer::Model>();
    
    auto embed = createLayer("embedding");
    embed->setProperty({"name=embed_tokens"});
    
    auto layers_0_norm = createLayer("layer_norm");
    layers_0_norm->setProperty({"name=layers_0_input_layernorm", "input_layers=embed_tokens"});
    
    // ... more layers ...
    
    // TODO(unsupported): scaled_dot_product_attention [attention] -- no direct mapping
    // Suggested approach: Fuse into neighboring matmul operations
    
    return model;
}

#ifdef NNTRAINER_STANDALONE_SMOKE_TEST
int main() {
    auto model = buildModel();
    model->compile();
    model->initialize();
    std::cout << "Model constructed OK" << std::endl;
    return 0;
}
#endif
```

### CAUSALLM_COMPONENT Output

```cpp
// generated_qwen3_causallm.h
#pragma once
#include <nntrainer/nntrainer.h>

namespace nntrainer {
namespace generated {

class Qwen3CausalLM {
public:
    void createLayer0_input_norm(Layer& layer);
    void createLayer0_attention(Layer& layer);
    // ... more methods ...
};

}  // namespace generated
}  // namespace nntrainer
```

---

## 10. CausalLM Component Installation

**File:** `engine/agents/causallm_install.py`

### Overview
Copies generated CausalLM component files to a real CausalLM project.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check header_path and source_path exist:                                 │
│     └─► If NOT ──► Return (nothing to install)                               │
│                                                                              │
│  2. Check state["install_generated_files"]:                                  │
│     └─► If FALSE ──► Log "disabled" ──► Return                               │
│                                                                              │
│  3. Check causallm_project_root configured and exists:                       │
│     └─► If NOT ──► Log error ──► Return                                      │
│                                                                              │
│  4. Create directories:                                                      │
│     ├─► header_dir = {project_root}/{generated_header_directory}             │
│     └─► source_dir = {project_root}/{generated_source_directory}             │
│                                                                              │
│  5. Copy files:                                                              │
│     ├─► shutil.copyfile(header_path, installed_header)                       │
│     └─► shutil.copyfile(source_path, installed_source)                       │
│                                                                              │
│  6. Update state with installed paths                                        │
│  7. Log reminder to add files to build system                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - Validates all preconditions
   - `os.makedirs(header_dir, exist_ok=True)`
   - `os.makedirs(source_dir, exist_ok=True)`
   - `shutil.copyfile(src, dst)` x2

### Preconditions (All Must Be True)

1. `state["causallm_header_path"]` and `state["causallm_source_path"]` exist
2. `state["install_generated_files"] == True`
3. `state["causallm_project_root"]` is set and directory exists

---

## 11. Compilation

**File:** `engine/agents/compiler_agent.py`

### Overview
Compiles generated C++ code against real nntrainer installation.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check emission_mode:                                                     │
│     └─► If "causallm_component" ──► _handle_causallm_component()            │
│         └─► Skip standalone compilation (not applicable)                     │
│                                                                              │
│  2. Check skip_cpp_compilation flag:                                         │
│     └─► If TRUE ──► Return (skipped)                                         │
│                                                                              │
│  3. Check cpp_path exists:                                                   │
│     └─► If NOT ──► Return (error: no file)                                   │
│                                                                              │
│  4. Find g++ compiler:                                                       │
│     └─► If NOT found ──► Return (skipped)                                    │
│                                                                              │
│  5. Discover nntrainer flags:                                                │
│     └─► cflags, libs, source = discover_flags()                              │
│     └─► If cflags is None ──► Return (nntrainer not found)                   │
│                                                                              │
│  6. Build compile command:                                                   │
│     └─► g++ -std=c++17 -DNNTRAINER_STANDALONE_SMOKE_TEST                    │
│         {cflags} {cpp_path} -o {binary_path} {libs}                          │
│                                                                              │
│  7. Run compilation:                                                         │
│     └─► subprocess.run(command, capture_output=True, text=True, timeout=180) │
│                                                                              │
│  8. Handle result:                                                           │
│     ├─► Success (returncode=0):                                              │
│     │   ├─► state["compiled"] = True                                         │
│     │   └─► state["binary_path"] = binary_path                               │
│     │                                                                        │
│     └─► Failure:                                                             │
│         ├─► state["compiled"] = False                                        │
│         └─► state["compile_log"] = stdout + stderr                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `_normalise_emission_mode(state)` - Handles enum/string conversion
   - `_handle_causallm_component(state)` (if applicable)
   - `discover_flags()` - Finds nntrainer include/lib paths
   - `subprocess.run(...)` - Executes g++

### Compile Command Template

```bash
g++ -std=c++17 -DNNTRAINER_STANDALONE_SMOKE_TEST \
    -I/path/to/nntrainer/include \
    /path/to/generated_model.cpp \
    -o /path/to/model_bin \
    -L/path/to/nntrainer/lib -lnntrainer
```

---

## 12. Auto-Fix

**File:** `engine/agents/auto_fix.py`

### Overview
Uses LLM to fix compilation errors (max 2 iterations).

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check prerequisites:                                                     │
│     └─► api_key AND cpp_code AND compile_log must exist                      │
│                                                                              │
│  2. Ensure langchain-anthropic installed:                                    │
│     └─► If NOT ──► pip install langchain-anthropic langchain-core            │
│                                                                              │
│  3. Build prompt:                                                            │
│     └─► "Fix ONLY what compiler errors point to..."                          │
│         ├─► Include compile_log[:4000]                                       │
│         └─► Include full cpp_code                                            │
│                                                                              │
│  4. Call LLM:                                                                │
│     └─► llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4000)      │
│     └─► resp = llm.invoke([HumanMessage(content=prompt)])                    │
│                                                                              │
│  5. Process response:                                                        │
│     ├─► Strip markdown fences if present                                     │
│     ├─► state["cpp_code"] = fixed                                            │
│     ├─► Write to state["cpp_path"]                                           │
│     ├─► state["fix_iterations"] += 1                                         │
│     └─► bus.code("generated_model.cpp", fixed)                               │
│                                                                              │
│  6. Return state (triggers recompile via LangGraph edge)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `_ensure_langchain_installed(api_key)`
   - `ChatAnthropic(model="claude-sonnet-4-6", ...)`
   - `llm.invoke([HumanMessage(content=prompt)])`
   - Writes fixed code to disk

### Retry Loop (LangGraph)

```
compiler ──► _should_retry_compile() ──► auto_fix ──► dual_graph ──► compiler
                │
                └─► If compiled=true OR fix_iterations>=2 ──► profiler
```

---

## 13. Profiling

**File:** `engine/agents/profiler_agent.py`

### Overview
Runs compiled binary and collects latency/memory metrics.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. Check compiled and binary_path:                                          │
│     └─► If NOT ──► state["profile"] = {"ran": False, "reason": ...}         │
│                                                                              │
│  2. Build command:                                                           │
│     └─► cmd = [binary_path]                                                  │
│     └─► If model_dir exists: cmd.append(model_dir)                           │
│                                                                              │
│  3. Run binary:                                                              │
│     └─► result = subprocess.run(cmd, capture_output=True, text=True,         │
│                                  timeout=120, check=False)                   │
│                                                                              │
│  4. Parse output:                                                            │
│     ├─► e2e_time: "[e2e time]: X ms"                                         │
│     ├─► peak_memory: "Peak memory usage (VmRSS): X KB"                       │
│     ├─► prefill: "prefill: X tokens, Y ms"                                   │
│     └─► generation: "generation: X tokens, Y ms"                             │
│                                                                              │
│  5. Estimate per-layer breakdown:                                            │
│     ├─► weighted_nodes = [n for n in nodes if status="mapped"]               │
│     ├─► total_weight = sum(attr_count for each node)                         │
│     └─► For each node: est_ms = total_ms * weight / total_weight             │
│                                                                              │
│  6. Build profile dict:                                                      │
│     └─► {                                                                    │
│         "ran": True,                                                         │
│         "total_latency_ms": ...,                                             │
│         "peak_memory_kb": ...,                                               │
│         "prefill_latency_ms": ...,                                           │
│         "generation_latency_ms": ...,                                        │
│         "layers": [...],                                                     │
│         "note": "per-layer figures are proportional estimates..."            │
│       }                                                                      │
│                                                                              │
│  7. bus.profile(profile)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `subprocess.run(cmd, ...)` - Executes binary
   - Regex parsing for metrics
   - Per-layer estimation based on attribute count

### Profile Output

```json
{
  "ran": true,
  "total_latency_ms": 125.432,
  "peak_memory_kb": 524288,
  "peak_memory_mb": 512.0,
  "prefill_latency_ms": 45.2,
  "generation_latency_ms": 80.2,
  "layers": [
    {"name": "embed_tokens", "type": "embedding", "estimated_ms": 2.5},
    {"name": "layers_0_self_attn_q_proj", "type": "fully_connected", "estimated_ms": 8.3},
    ...
  ],
  "note": "per-layer figures are proportional estimates from op weight, not a traced measurement"
}
```

---

## 14. Artifact Management

**File:** `engine/agents/artifact_manager.py`

### Overview
Walks output directory and collects all generated files for webview display.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run(state)                                          │
│                                                                              │
│  1. out_dir = state["out_dir"]                                               │
│                                                                              │
│  2. Walk directory tree:                                                     │
│     └─► For each file:                                                       │
│         ├─► Calculate relative path                                          │
│         ├─► Get stat (size, mtime)                                           │
│         ├─► Determine type by extension                                      │
│         └─► Append to items list                                             │
│                                                                              │
│  3. Sort items by path                                                       │
│  4. state["artifacts"] = items                                               │
│  5. bus.artifacts(items)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Function Call Sequence

1. `run(state: dict) -> dict`
   - `os.walk(out_dir)` - Recursive directory traversal
   - `os.stat(full_path)` - File metadata
   - `_kind(name)` - Type classification by extension

### Artifact Types

| Extension | Type |
|-----------|------|
| `.cpp` | C++ Source |
| `.h` | C++ Header |
| `.ini` | INI File |
| `.json` | JSON File |
| `.bin` | Binary File |
| `.safetensors` | Weights |
| `.log` | Log File |
| `.txt` | Text File |

---

## 15. Chat Agent

**File:** `engine/agents/chat_agent.py`

### Overview
Provides conversational interface for querying pipeline results using LangChain agents.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    run(out_dir, message, api_key, ums_token)                │
│                                                                              │
│  1. Load state: state = _load_state(out_dir)                                 │
│     └─► Reads state.json from disk                                           │
│                                                                              │
│  2. Check API key:                                                           │
│     └─► If NOT (api_key OR ums_token) ──► Log error ──► Return               │
│                                                                              │
│  3. Check state exists:                                                      │
│     └─► If empty ──► "No pipeline has been run yet" ──► Return               │
│                                                                              │
│  4. Ensure langchain installed:                                              │
│     └─► pip install langchain langchain-anthropic langchain-openai          │
│                                                                              │
│  5. Create tools: _make_tools(state)                                         │
│     ├─► get_model_summary()                                                  │
│     ├─► list_unsupported_ops()                                               │
│     ├─► get_artifacts()                                                      │
│     ├─► get_profile()                                                        │
│     ├─► get_graph_nodes_by_type(node_type)                                   │
│     └─► get_decoder_layer_structure(layer_index)                             │
│                                                                              │
│  6. Initialize LLM:                                                          │
│     ├─► If ums_token: ChatOpenAI(api_base="http://localhost:6543/v1")        │
│     └─► Else: ChatAnthropic(model="claude-sonnet-4-6")                       │
│                                                                              │
│  7. Create agent:                                                            │
│     ├─► prompt = ChatPromptTemplate.from_messages(...)                       │
│     ├─► agent = create_tool_calling_agent(llm, tools, prompt)                │
│     └─► executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6) │
│                                                                              │
│  8. Run: result = executor.invoke({"input": message})                        │
│  9. bus.chat("assistant", result["output"])                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Available Tools

| Tool | Description |
|------|-------------|
| `get_model_summary()` | Returns architecture and compatibility summary |
| `list_unsupported_ops()` | Lists unsupported ops with reasons and suggestions |
| `get_artifacts()` | Lists all generated files with sizes |
| `get_profile()` | Returns profiling results |
| `get_graph_nodes_by_type(type)` | Filters graph nodes by type |
| `get_decoder_layer_structure(index)` | Shows detailed layer structure |

---

## Event Bus Communication

All agents communicate with the webview via the centralized event bus (`engine/agents/events.py`).

### Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `agent_status` | Agent progress updates | `{agent, status, detail}` |
| `log` | Log messages | `{message, level}` |
| `graph` | Graph data | `{nodes, edges, target}` |
| `node_mappings` | Cross-highlight mappings | `[{sourceIds, targetIds, ...}]` |
| `code` | Code content | `{filename, content}` |
| `file_content` | File content for export | `{tab, filename, content}` |
| `profile` | Profiling results | Profile dict |
| `artifacts` | File listing | `[{path, type, size, modified}]` |
| `pipeline_complete` | Pipeline finished | Summary dict |
| `chat` | Chat messages | `{role, content}` |
| `error` | Error messages | `{source, message}` |

---

## Appendix A: PyTorch to nntrainer IR Conversion Flow

This section documents the complete Intermediate Representation (IR) conversion flow from PyTorch models to nntrainer graphs.

### Overview

The conversion follows a **three-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PyTorch to nntrainer Conversion Pipeline                  │
│                                                                              │
│  Stage 1: PyTorch Module Tree                                                │
│            (AutoModel.from_config)                                           │
│                    │                                                         │
│                    ▼                                                         │
│  Stage 2: Semantic IR (CausalLMIR)                                           │
│            (Architecture-agnostic, framework-neutral)                        │
│                    │                                                         │
│                    ▼                                                         │
│  Stage 3: nntrainer Graph                                                    │
│            (Target-specific, C++ generator input)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: PyTorch Module Tree → Semantic IR

**File:** `api/adapters/llama_family.py` (LlamaFamilyAdapter)

#### Function: `build_semantic_ir(config, model)`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LlamaFamilyAdapter.build_semantic_ir()                    │
│                                                                              │
│  Input:                                                                      │
│    - config: HuggingFace AutoConfig (from config.json)                      │
│    - model: AutoModel.from_config(config) - module tree with random weights │
│                                                                              │
│  Process:                                                                    │
│    1. Extract global config:                                                 │
│       ├─► hidden_size                                                        │
│       ├─► num_hidden_layers (num_layers)                                     │
│       ├─► vocab_size                                                         │
│       ├─► rms_norm_eps                                                       │
│       └─► tie_word_embeddings                                                │
│                                                                              │
│    2. For each decoder layer (model.layers[i]):                              │
│       └─► _build_decoder_layer(i, layer, config, hidden_size, eps)           │
│           ├─► _build_attention(...) ──► AttentionIR                          │
│           ├─► _rms_norm(...) ──► input_norm (NormIR)                         │
│           ├─► _rms_norm(...) ──► post_attention_norm (NormIR)                │
│           └─► _build_mlp(...) ──► MLPIR                                      │
│                                                                              │
│    3. Build final norm:                                                      │
│       └─► _rms_norm("model.norm", model.norm, hidden_size, eps)              │
│                                                                              │
│  Output: CausalLMIR dataclass                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Semantic IR Data Structures (`api/semantic/model.py`)

```python
@dataclass
class CausalLMIR:
    architecture: str           # e.g., "llama", "qwen2", "qwen3"
    hidden_size: int            # e.g., 4096
    vocab_size: int             # e.g., 32000
    num_layers: int             # e.g., 32
    embedding_name: str         # "model.embed_tokens"
    decoder_layers: list[DecoderLayerIR]
    final_norm: NormIR
    lm_head_name: str           # "lm_head"
    tied_embeddings: bool       # True if weights are shared

@dataclass
class DecoderLayerIR:
    index: int
    input_norm: NormIR
    attention: AttentionIR
    post_attention_norm: NormIR
    mlp: MLPIR

@dataclass
class AttentionIR:
    source_name: str
    q_proj: ProjectionIR
    k_proj: ProjectionIR
    v_proj: ProjectionIR
    o_proj: ProjectionIR
    num_heads: int
    num_kv_heads: int           # For GQA/MQA
    head_dim: int
    q_norm: Optional[NormIR]    # Qwen3-style per-head norm
    k_norm: Optional[NormIR]
    rope_theta: float
    max_position_embeddings: int
    sliding_window: Optional[int]
    causal: bool
    use_kv_cache: bool

@dataclass
class MLPIR:
    source_name: str
    up_proj: ProjectionIR
    down_proj: ProjectionIR
    activation: str             # "silu", "gelu", etc.
    gated: bool                 # True if gate_proj exists
    gate_proj: Optional[ProjectionIR]

@dataclass
class ProjectionIR:
    source_name: str
    output_size: int
    bias: bool
    weight_name: str
    input_size: Optional[int]

@dataclass
class NormIR:
    source_name: str
    norm_type: str              # "rms_norm"
    feature_size: int
    epsilon: float
    reshaped: bool              # True for per-head norms
```

#### Adapter Selection (`api/adapters/registry.py`)

```python
def select_adapter(config, model) -> Optional[ArchitectureAdapter]:
    model_type = getattr(config, "model_type", None)
    
    # Match model_type to registered adapters:
    # - "llama" → LlamaAdapter (extends LlamaFamilyAdapter)
    # - "mistral" → MistralAdapter (extends LlamaFamilyAdapter)
    # - "qwen2" → Qwen2Adapter (extends LlamaFamilyAdapter)
    # - "qwen3" → Qwen3Adapter (extends LlamaFamilyAdapter)
    
    return adapter_class() if matched else None
```

### Stage 2: Semantic IR → nntrainer Graph

**File:** `api/lowering/nntrainer/lowerer.py` (NNTrainerLowerer)

#### Function: `lower()`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NNTrainerLowerer.lower()                            │
│                                                                              │
│  Input: CausalLMIR from Stage 1                                              │
│                                                                              │
│  Process:                                                                    │
│    1. Create Graph object                                                    │
│    2. Set metadata:                                                          │
│       ├─► emission_mode = "causallm_component"                               │
│       ├─► uniform_layers = (all layers structurally identical?)              │
│       └─► architecture, num_layers                                           │
│                                                                              │
│    3. Build graph sequentially:                                              │
│       │                                                                      │
│       ├─► hidden = _lower_embedding(builder)                                 │
│       │   └─► GraphNode: embedding layer                                     │
│       │       - node_type: "embedding"                                       │
│       │       - semantic_type: "embedding"                                   │
│       │       - attributes: {out_dim: hidden_size}                           │
│       │       - weight_name: "{embedding_name}.weight"                       │
│       │                                                                      │
│       ├─► For each decoder_layer in model_ir.decoder_layers:                 │
│       │   └─► hidden = _lower_decoder_layer(builder, layer, hidden)          │
│       │       │                                                              │
│       │       ├─► normed = _lower_norm(input_norm, hidden)                   │
│       │       │   └─► rms_norm or reshaped_rms_norm                          │
│       │       │                                                              │
│       │       ├─► attn_out = _lower_attention(attn, normed)                  │
│       │       │   ├─► q = _lower_projection(q_proj, normed)                  │
│       │       │   ├─► k = _lower_projection(k_proj, normed)                  │
│       │       │   ├─► v = _lower_projection(v_proj, normed)                  │
│       │       │   ├─► (Optional) q_norm, k_norm                              │
│       │       │   ├─► kv_cache_placeholders                                  │
│       │       │   ├─► mha_core (fused attention)                             │
│       │       │   └─► o_proj = _lower_projection(o_proj, mha_core)           │
│       │       │                                                              │
│       │       ├─► after_attn = _lower_residual_add(hidden, attn_out)         │
│       │       │   └─► addition node with two inputs                          │
│       │       │                                                              │
│       │       ├─► post_normed = _lower_norm(post_attention_norm, after_attn) │
│       │       │                                                              │
│       │       ├─► mlp_out = _lower_mlp(mlp, post_normed)                     │
│       │       │   ├─► If gated:                                              │
│       │       │   │   ├─► gate = _lower_projection(gate_proj)                │
│       │       │   │   ├─► up = _lower_projection(up_proj)                    │
│       │       │   │   ├─► activation (on gate)                               │
│       │       │   │   ├─► multiply (activation * up)                         │
│       │       │   │   └─► down_proj                                          │
│       │       │   └─► Else:                                                  │
│       │       │       ├─► up_proj                                            │
│       │       │       ├─► activation                                         │
│       │       │       └─► down_proj                                          │
│       │       │                                                              │
│       │       └─► after_mlp = _lower_residual_add(after_attn, mlp_out)       │
│       │                                                                      │
│       ├─► hidden = _lower_final_norm(hidden)                                 │
│       │   └─► rms_norm for final normalization                               │
│       │                                                                      │
│       └─► _lower_lm_head(hidden)                                             │
│           └─► fully_connected layer to vocab_size                            │
│               (tied_embeddings? use embedding weight : own weight)           │
│                                                                              │
│  Output: Graph (api.graph.graph.Graph)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### nntrainer Graph Node Structure

```python
class GraphNode:
    id: str                       # Unique identifier
    name: str                     # Human-readable name
    node_type: str                # nntrainer layer type
    semantic_type: str            # Semantic role (attention, mlp, etc.)
    group_id: str                 # For grouping (e.g., "decoder_0")
    template_id: str              # For codegen reuse
    status: str                   # "supported", "fused", "host_managed"
    supported: bool
    attributes: dict              # Layer-specific properties
    weight_name: str              # Weight tensor name
    source_node_ids: list[str]    # Original PyTorch module names
    inputs: list[str]             # Predecessor node IDs
    outputs: list[str]            # Successor node IDs
```

#### nntrainer Layer Types Mapping

| PyTorch Module | nntrainer Layer Type | Notes |
|----------------|---------------------|-------|
| `nn.Embedding` | `embedding` | Input token embedding |
| `nn.Linear` (Q/K/V) | `fully_connected` | Attention projections |
| `nn.Linear` (O) | `fully_connected` | Output projection |
| `nn.Linear` (MLP) | `fully_connected` | up/down/gate projections |
| `RMSNorm` | `rms_norm` | Standard normalization |
| `RMSNorm` (per-head) | `reshaped_rms_norm` | Q/K norms with reshaped input |
| Attention (fused) | `mha_core` | Multi-head attention with RoPE |
| Residual add | `addition` | Skip connection |
| SiLU/GELU activation | `activation` | MLP activation |
| Gated multiply | `multiply` | Gate × Up projection |
| KV Cache | `kv_cache_placeholders` | Host-managed cache |

### Stage 3: nntrainer Graph → C++ Code

**File:** `converters/cpp_generator.py` (CPPGenerator)

#### Function: `generate_component()`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CPPGenerator.generate_component()                         │
│                                                                              │
│  Input: Graph from NNTrainerLowerer                                          │
│                                                                              │
│  Process:                                                                    │
│    1. Check uniform_layer_signature() - can layers share one function?       │
│    2. Generate header file:                                                  │
│       ├─► Class declaration: {Architecture}CausalLM                          │
│       ├─► Method declarations per layer type                                 │
│       └─► Member variables for layers                                        │
│                                                                              │
│    3. Generate source file:                                                  │
│       ├─► createLayer() methods for each template_id                         │
│       ├─► setProperty() calls for attributes                                 │
│       └─► Weight loading comments                                          │
│                                                                              │
│  Output: GeneratedFiles(header, source, architecture, ...)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complete Conversion Example

#### PyTorch Input (Llama-2-7B)

```python
# config.json
{
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "rms_norm_eps": 1e-6,
    "vocab_size": 32000
}

# Module tree (AutoModel.from_config)
LlamaForCausalLM(
    model: LlamaModel(
        embed_tokens: Embedding(32000, 4096)
        layers: ModuleList([
            LlamaDecoderLayer(
                self_attn: LlamaAttention(
                    q_proj: Linear(4096, 4096)
                    k_proj: Linear(4096, 4096)
                    v_proj: Linear(4096, 4096)
                    o_proj: Linear(4096, 4096)
                )
                mlp: LlamaMLP(
                    gate_proj: Linear(4096, 11008)
                    up_proj: Linear(4096, 11008)
                    down_proj: Linear(11008, 4096)
                )
                input_layernorm: LlamaRMSNorm(4096)
                post_attention_layernorm: LlamaRMSNorm(4096)
            )
        ] * 32)
        norm: LlamaRMSNorm(4096)
    )
    lm_head: Linear(4096, 32000)
)
```

#### Semantic IR (CausalLMIR)

```python
CausalLMIR(
    architecture="llama",
    hidden_size=4096,
    vocab_size=32000,
    num_layers=32,
    embedding_name="model.embed_tokens",
    decoder_layers=[
        DecoderLayerIR(
            index=0,
            input_norm=NormIR("model.layers.0.input_layernorm", "rms_norm", 4096, 1e-6),
            attention=AttentionIR(
                source_name="model.layers.0.self_attn",
                q_proj=ProjectionIR("model.layers.0.self_attn.q_proj", 4096, 4096),
                k_proj=ProjectionIR("model.layers.0.self_attn.k_proj", 4096, 4096),
                v_proj=ProjectionIR("model.layers.0.self_attn.v_proj", 4096, 4096),
                o_proj=ProjectionIR("model.layers.0.self_attn.o_proj", 4096, 4096),
                num_heads=32, num_kv_heads=32, head_dim=128,
                rope_theta=10000.0, causal=True
            ),
            post_attention_norm=NormIR(...),
            mlp=MLPIR(
                source_name="model.layers.0.mlp",
                up_proj=ProjectionIR(..., 11008),
                down_proj=ProjectionIR(..., 4096),
                gate_proj=ProjectionIR(..., 11008),
                gated=True, activation="silu"
            )
        )
    ] * 32,
    final_norm=NormIR("model.norm", "rms_norm", 4096, 1e-6),
    lm_head_name="lm_head",
    tied_embeddings=False
)
```

#### nntrainer Graph (partial)

```
Nodes:
  - embedding (embedding)
      → attributes: {out_dim: 4096}
      → weight: model.embed_tokens.weight [32000, 4096]
  
  - decoder_0.input_norm (rms_norm)
      → attributes: {epsilon: 1e-6, feature_size: 4096}
      → weight: model.layers.0.input_layernorm.weight [4096]
  
  - decoder_0.wq (fully_connected)
      → attributes: {unit: 4096, disable_bias: true}
      → weight: model.layers.0.self_attn.q_proj.weight [4096, 4096]
  
  - decoder_0.wk (fully_connected)
  - decoder_0.wv (fully_connected)
  
  - decoder_0.mha_core (mha_core)
      → attributes: {num_heads: 32, num_heads_kv: 32, rope_theta: 10000.0, is_causal: true}
  
  - decoder_0.wo (fully_connected)
  
  - decoder_0.attention_residual (addition)
      → inputs: [decoder_0.input_norm, decoder_0.wo]
  
  ... (repeated for 32 layers)
  
  - final_norm (rms_norm)
  - lm_head (fully_connected)
      → attributes: {unit: 32000}
      → weight: lm_head.weight [32000, 4096]
```

### Key Design Decisions

1. **No Direct torch.fx Tracing**: The pipeline uses module-tree walking instead of torch.fx symbolic tracing to handle data-dependent control flow (e.g., `if` statements in transformers).

2. **Architecture-Agnostic Semantic IR**: The `CausalLMIR` is framework-neutral, allowing the same lowering logic to work for Llama, Mistral, Qwen2, Qwen3, etc.

3. **Runtime Feature Detection**: The `LlamaFamilyAdapter` detects features like Q/K norms and sliding windows at runtime from the actual module tree, not from hardcoded subclass logic.

4. **Structural Signature for Codegen**: The `uniform_layer_signature()` method determines if all decoder layers are structurally identical, enabling loop-based codegen instead of unrolled code.

5. **Thread-Safe Graph Building**: The `ThreadSafeGraphBuilder` allows concurrent construction of independent branches (Q/K/V projections) while maintaining graph consistency.

---

## Appendix B: File Structure

```
extension/
├── engine/
│   ├── orchestrator_main.py      # Entry point
│   ├── agents/
│   │   ├── orchestrator.py       # Pipeline coordination
│   │   ├── model_discovery.py    # HF config extraction
│   │   ├── compatibility.py      # Op compatibility check
│   │   ├── weight_download.py    # Weight download
│   │   ├── weight_converter.py   # Weight format conversion
│   │   ├── nntrainer_lowering.py # Semantic to nntrainer
│   │   ├── ini_generator.py      # INI file generation
│   │   ├── graph_builder.py      # Internal graph view
│   │   ├── graph_views.py        # Webview graph building
│   │   ├── dual_graph.py         # Two-graph publishing
│   │   ├── cpp_generator_agent.py # C++ code generation
│   │   ├── causallm_install.py   # CausalLM file install
│   │   ├── compiler_agent.py     # C++ compilation
│   │   ├── auto_fix.py           # LLM-based fix
│   │   ├── profiler_agent.py     # Latency profiling
│   │   ├── artifact_manager.py   # File collection
│   │   ├── chat_agent.py         # Chat interface
│   │   ├── state.py              # PipelineState type
│   │   ├── events.py             # Event bus
│   │   └── cache.py              # 30-day caching
│   └── api/
│       ├── adapters/             # Model architecture adapters
│       │   ├── base.py           # ArchitectureAdapter base class
│       │   ├── registry.py       # Adapter selection
│       │   ├── llama.py          # Llama adapter
│       │   ├── llama_family.py   # LlamaFamilyAdapter (shared)
│       │   ├── mistral.py        # Mistral adapter
│       │   └── qwen2.py, qwen3.py
│       ├── semantic/
│       │   └── model.py          # CausalLMIR dataclasses
│       ├── lowering/
│       │   └── nntrainer/
│       │       ├── lowerer.py    # NNTrainerLowerer
│       │       ├── validation.py # Graph validation
│       │       ├── manifest.py   # Weight/config manifests
│       │       └── concurrency.py # ThreadSafeGraphBuilder
│       ├── compatibility/
│       │   ├── op_table.py       # nntrainer op compatibility table
│       │   ├── op_level_checker.py
│       │   └── semantic_capabilities.py
│       ├── graph/
│       │   ├── graph.py          # Graph data structure
│       │   ├── node.py           # GraphNode
│       │   └── edge.py           # GraphEdge
│       └── parsers/
│           ├── generic_fx_parser.py  # Module-tree walker
│           └── edge_resolver.py
├── webview/
│   └── main.html                 # Single-file UI
├── EXECUTION_FLOW.md             # High-level flow
├── FEATURE_EXECUTION_FLOWS.md    # This document
└── DECISIONS.md                  # Architecture decisions
```
