# Execution Flow Map

## Overview

This document describes the complete execution flow of the AI Compiler Workbench extension, from user initiating a pipeline to final artifact generation.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Clicks "Run Pipeline"                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Orchestrator.run_pipeline()                                              │
│     - Clears weight cache from previous runs                                 │
│     - Creates new PipelineState                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Model Discovery Agent                                                    │
│     - Downloads model config from HuggingFace                                │
│     - Extracts architecture, hidden_size, vocab_size, num_layers             │
│     - Stores config in state["hf_config"]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────┐
│  3a. Weight Download (Background) │    │  3b. Compatibility Agent             │
│     - Downloads safetensors       │    │     - Builds graph from config only  │
│     - Stores in weights_path      │    │     - No weight download dependency  │
└──────────────────────────────────┘    └──────────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. NNTrainer Lowering Agent                                                 │
│     - Converts semantic IR to nntrainer graph                                │
│     - Builds weight preview cache from downloaded weights                    │
│     - Creates model_graph_view and nntrainer_graph_view                      │
│     - Generates node_mappings for cross-highlighting                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────┐
│  5a. INI Generator Agent           │    │  5b. Graph Builder Agent             │
│     - Generates model.ini          │    │     - Validates graph structure      │
│     - From nntrainer_graph_ir      │    │     - Adds compatibility suggestions │
└──────────────────────────────────┘    └──────────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. C++ Generator Agent                                                      │
│     - Generates C++ code from graph IR                                       │
│     - Two modes: MODEL_API or CAUSALLM_COMPONENT                             │
│     - Outputs generated_model.cpp                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. CausalLM Install Agent                                                   │
│     - Copies generated files to CausalLM project                             │
│     - Installs header and source files                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. CausalLM Build & Run Agent                                               │
│     - Builds CausalLM project with generated code                            │
│     - Runs smoke test inference                                              │
│     - Sets compiled=true on success                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  9. Dual Graph Agent                                                         │
│     - Publishes Model Graph and nntrainer Graph to webview                   │
│     - Sends node_mappings for cross-highlighting                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  10. Compiler Agent                                                          │
│     - Compiles generated C++ code                                            │
│     - Sets compiled=true on success                                          │
│     - Sets compile_log with output                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          ┌─────────▼─────────┐             ┌──────────▼──────────┐
          │  Success          │             │  Failure            │
          │  (compiled=true)  │             │  (compiled=false)   │
          └─────────┬─────────┘             └──────────┬──────────┘
                    │                                   │
                    │                    ┌──────────────▼──────────────┐
                    │                    │  Auto-Fix Agent (max 2x)    │
                    │                    │     - Analyzes error        │
                    │                    │     - Suggests fixes        │
                    │                    └──────────────┬──────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  11. Profiler Agent                                                          │
│      - Runs smoke test profiling                                             │
│      - Collects layer latencies                                              │
│      - Identifies bottlenecks                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  12. Join Weight Download                                                    │
│      - Waits for background weight download to complete                      │
│      - Merges weights_path into state                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  13. Artifact Manager                                                        │
│      - Collects all generated files                                          │
│      - Lists in webview Artifacts panel                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  14. Pipeline Complete                                                       │
│      - Saves state.json                                                      │
│      - Sends summary to webview                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## State Flow

The `PipelineState` dictionary flows through all agents, with each agent adding/modifying specific keys:

| Key | Set By | Description |
|-----|--------|-------------|
| `model_name` | User | Model identifier (e.g., "meta-llama/Llama-2-7b") |
| `out_dir` | User | Output directory for generated files |
| `hf_config` | model_discovery | HuggingFace config dict |
| `architecture` | model_discovery | Model architecture name |
| `semantic_ir` | compatibility | Semantic model IR as dict |
| `weights_path` | weight_download | Path to downloaded safetensors |
| `converted_weights_path` | weight_converter | Path to converted binary weights |
| `nntrainer_graph_ir` | nntrainer_lowering | Lowered nntrainer graph |
| `model_graph_view` | nntrainer_lowering | Model graph for webview |
| `nntrainer_graph_view` | nntrainer_lowering | nntrainer graph for webview |
| `node_mappings` | nntrainer_lowering | Source<->target node mappings |
| `ini_content` | ini_generator | Generated model.ini content |
| `cpp_code` | cpp_generator | Generated C++ code |
| `compiled` | compiler_agent | true if compilation succeeded |
| `compile_log` | compiler_agent | Compiler output |
| `profile` | profiler_agent | Profiling results |
| `artifacts` | artifact_manager | List of generated files |

## Event Flow to Webview

```
Backend (Python)                    VSCode Extension              Webview (HTML/JS)
      │                                   │                              │
      │─── agent_status ──────────────────│──────────────────────────────►│
      │                                   │                              │
      │─── log ───────────────────────────│──────────────────────────────►│
      │                                   │                              │
      │─── graph (target="model") ────────│──────────────────────────────►│
      │                                   │         Renders Model Graph  │
      │                                   │                              │
      │─── graph (target="nntrainer") ────│──────────────────────────────►│
      │                                   │      Renders nntrainer Graph │
      │                                   │                              │
      │─── node_mappings ─────────────────│──────────────────────────────►│
      │                                   │   Enables cross-highlighting │
      │                                   │                              │
      │─── code ──────────────────────────│──────────────────────────────►│
      │                                   │        Updates C++ viewer    │
      │                                   │                              │
      │─── file_content ──────────────────│──────────────────────────────►│
      │                                   │      Enables file export     │
      │                                   │                              │
      │─── profile ───────────────────────│──────────────────────────────►│
      │                                   │       Updates profiler UI    │
      │                                   │                              │
      │─── artifacts ─────────────────────│──────────────────────────────►│
      │                                   │      Lists artifacts panel   │
      │                                   │                              │
      │─── pipeline_complete ─────────────│──────────────────────────────►│
      │                                   │      Shows summary status    │
      │                                   │                              │
```

## Weight Preview Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. nntrainer_lowering.py                                                    │
│     - Calls build_weight_cache(weights_path)                                 │
│     - Reads all tensors from safetensors files                               │
│     - Extracts: preview (first 5 values), min, max, mean, std                │
│     - Calls graph_views.set_weight_cache(cache)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. build_nntrainer_graph_view()                                             │
│     - For each node with weight_name:                                        │
│       - Calls get_weight_preview(weight_name)                                │
│       - Merges preview data into weightInfo                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. Webview Inspector                                                        │
│     - User clicks node in graph                                              │
│     - renderInspector(n) displays weightInfo                                 │
│     - Shows: name, shape, dtype, params, statistics, preview values          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cleanup on New Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. run_pipeline()                                                           │
│     - Calls graph_views.clear_weight_cache()                                 │
│     - Creates fresh PipelineState                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Webview receives pipelineStarted                                         │
│     - Clears fileContents                                                    │
│     - Resets graph panes                                                     │
│     - Clears agent status                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files and Their Responsibilities

| File | Responsibility |
|------|---------------|
| `orchestrator.py` | Coordinates all agents, manages pipeline flow |
| `model_discovery.py` | Fetches model config from HuggingFace |
| `compatibility.py` | Builds semantic graph from config |
| `nntrainer_lowering.py` | Converts semantic IR to nntrainer, builds weight cache |
| `weight_download.py` | Downloads model weights (background) |
| `weight_converter.py` | Converts weights to binary format |
| `ini_generator.py` | Generates model.ini file |
| `graph_builder.py` | Validates and enhances graph |
| `cpp_generator.py` | Generates C++ code |
| `graph_views.py` | Builds graph views for webview |
| `dual_graph.py` | Publishes graphs to webview |
| `compiler_agent.py` | Compiles generated C++ |
| `auto_fix.py` | Suggests fixes for compile errors |
| `profiler_agent.py` | Profiles model execution |
| `artifact_manager.py` | Collects generated files |
| `main.html` | Webview UI (graphs, chat, logs, profiler) |
