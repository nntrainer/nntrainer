# AI Compiler Workbench -- Agentic Graph Visualiser

Converts a HuggingFace model into nntrainer-compatible C++ and an interactive
knowledge graph, using a multi-agent pipeline orchestrated with **LangGraph**
(agent nodes over a shared state, with a bounded compile/auto-fix loop) and
**LangChain** (the two LLM-optional agents, plus the chat panel).

This replaces the earlier single-script version of this extension. The
architecture follows the "Agentic Graph Visualiser" design: an Orchestrator
Agent coordinates a fixed pipeline of specialist agents, most of which are
deterministic (no LLM) and a couple of which use an LLM only where reasoning
genuinely helps.

## Agent pipeline

| Agent | LLM? | What it does |
|---|---|---|
| Model Discovery | No | Pulls `config.json` / architecture metadata from the HF Hub |
| Compatibility | Optional | Traces the model with `torch.fx`, checks every op against the nntrainer op table, asks Claude for a short suggestion on any unsupported op |
| Weight Download | No | Downloads `.safetensors`/`.bin` weights via `huggingface_hub` |
| Weight Converter | No | Flattens the HF tensors into nntrainer's binary weight layout |
| INI Generator | No | Emits `model.ini` (nntrainer's alternative construction path) |
| Graph Builder / Visualization | No | Lays out the traced IR into the node/edge graph the webview renders |
| C++ Generator | LLM-assisted | Emits `generated_model.cpp` (`createModel()` + a guarded smoke-test `main()`); unmapped ops stay visible `TODO`s, annotated with the Compatibility Agent's suggestion -- never silently guessed |
| Compiler | No | Compiles against a real nntrainer install if one is configured; otherwise reports "not compiled" honestly instead of faking success |
| Auto-Fix | LLM | Only runs on a **real g++ error** (not on unsupported-op TODOs), patches the file, bounded to 2 attempts |
| Profiler | No | Runs the compiled binary and reports wall-clock latency; per-layer numbers are a clearly-labeled proportional estimate, not a traced measurement |
| Artifact Manager | No | Walks the output directory and reports every file produced |

Progress streams live into the webview (Agent Pipeline list, Logs, Model
Graph, C++ Code Viewer, Profiler Dashboard, Artifacts) as one JSON event per
line over the child process's stdout -- there's no server/websocket, just an
honest pipe from a local Python process into the extension host.

## Setup

1. Python 3.10+ on your PATH (or set `aiCompilerWorkbench.pythonPath`).
2. Run **AI Compiler Workbench: Check / Install Python Dependencies**, or
   manually: `pip install -r engine/requirements.txt`.
3. (Optional) Set `aiCompilerWorkbench.anthropicApiKey` in Settings to enable
   the Compatibility Agent's suggestions, the Auto-Fix Agent, and the Chat
   panel. Everything else works without it.
4. (Optional) To let the Compiler/Profiler agents actually build and run the
   generated file against a real nntrainer install, set the
   `NNTRAINER_INCLUDE_DIR` (and optionally `NNTRAINER_LIB_DIR`) environment
   variables before launching VS Code. Without these, those two agents report
   "not compiled -- nntrainer not found" instead of pretending to succeed.

## Commands

- **AI Compiler Workbench: Open Agentic Graph Visualiser** -- opens the panel.
- **AI Compiler Workbench: Run Pipeline (Convert HuggingFace Model)** -- runs
  the full agent pipeline for a model you name (HF Hub ID or local path).
- **AI Compiler Workbench: Stop Pipeline**
- **AI Compiler Workbench: Check / Install Python Dependencies**

## Profile On-Device (real hardware, real nntrainer)

A separate, explicit action -- **AI Compiler Workbench: Profile On-Device**
(or the "🔧 Profile On-Device" button) -- that:

1. Prompts you to locate a local **nntrainer** install if
   `aiCompilerWorkbench.nntrainerPath` isn't set yet (a folder containing
   `include/nntrainer` and `lib/`, built from
   [nntrainer/nntrainer](https://github.com/nntrainer/nntrainer) via:
   ```
   git clone https://github.com/nntrainer/nntrainer.git
   cd nntrainer
   meson --prefix=<install-dir> build
   ninja -C build install
   ```
   This builds **natively for whatever machine you run it on** (x86_64 or
   arm64, auto-detected via `platform.machine()`) -- there's no cross-compile
   toolchain bundled here; profiling an arm64 target from an x86_64 host (or
   vice versa) needs your own cross toolchain and is out of scope.

2. Wraps the already-generated `createModel()` in a real, runnable harness:
   construct -> compile -> initialize -> **load the real converted
   weights** -> 3 warm-up + 10 timed `forwarding()` calls -> report
   avg/min/max wall-clock latency, genuinely measured on your machine.
   Model *construction* reuses the exact same `createLayer()`/`addLayer()`/
   `NeuralNetwork` pattern already used throughout nntrainer's own
   `Applications/*` (including `Applications/CausalLM`); only the weight-load
   and timed-forwarding calls are new, and they're isolated in one clearly
   commented block in `profile/profiling_harness.cpp` so that if your
   installed nntrainer version's exact `load()`/`forwarding()` signature
   differs, the compiler error points at exactly that block.

3. Reports **likely bottleneck layers**: the real measured average latency
   distributed across layers proportional to each layer's real parameter
   count (captured from the model's actual `nn.Module` parameters during
   tracing) -- a defensible compute-share proxy, clearly labeled as an
   estimate, not a per-layer trace. nntrainer's public API doesn't expose a
   documented, stable "time this one layer" hook to measure it more directly
   without risking exactly the kind of confidently-wrong code this tool
   exists to avoid.

Results land in `nntrainer_out/profile/profile_report.json` and show up
live in the Profiler Dashboard (avg latency, target arch, whether real
weights were loaded, and a ranked bottleneck-layer list).

## Two graphs, side by side, both real, both exportable

Both graphs are shown **at the same time**, side by side, each parsed
directly from the actual generated file (not the same in-memory data
shown twice):

- **Model Graph (.ini)** -- parsed straight out of `model.ini`, including the
  `input_layers = ...` property the INI Generator Agent writes for every
  node based on its *real* predecessors in the traced graph (resolved through
  any unsupported/passthrough nodes to the nearest real ancestor), and a
  `; weight: name=... shape=[...] dtype=... params=...` comment per node.
- **C++ Graph** -- parsed straight out of `generated_model.cpp`'s
  `createLayer()` / `setProperty({"input_layers=..."})` calls and its
  `// Weight: name=... shape=[...] dtype=... params=...` comments, plus any
  `TODO(unsupported)` blocks as red "unmapped" nodes.

Because both are parsed from the files actually written to disk, they double
as a cross-check that the `.ini` and `.cpp` agree with each other. Each pane
has its own **Export .ini** / **Export .cpp** button that saves that file
directly via a native save dialog.

## Inspector panel

Clicking any node in either graph opens an **Inspector** (same pattern as the
reference nntrainer graph visualizer) docked in the top-right of that graph
pane, with two tabs:
- **Properties** -- type, status, resolved input/output shape, layer attributes
- **Weights** -- the real weight tensor's name, shape, dtype, and parameter
  count, captured directly from the model's `nn.Module` parameters during
  tracing (or "No weight data" if the node has none, e.g. LayerNorm-free
  activations or unsupported ops without a weight tensor).

## Graph UI controls

- **Vertical, layered layout**: one row per topological depth, top to bottom.
- **Grab-to-pan**: click-drag anywhere on a graph's background to scroll it (toggle per-pane with its "✋ Pan" button).
- **Minimap**: bottom-right overview per graph with a draggable viewport indicator -- click or drag on it to jump around a large graph.
- **Fit**: resets that graph's scroll to the top-left origin.
- Independent pan/zoom/minimap state per pane -- panning the .ini graph doesn't affect the C++ graph.

## Fully resizable layout

Every panel can be resized by dragging the splitter next to it: Agent
Pipeline, both graph panes, the C++ Code Viewer, and (via the horizontal
splitter between the top and bottom sections) Chat, Logs, Profiler Dashboard,
and Artifacts. Dragging a splitter only resizes the two panels touching it,
so the rest of the layout stays put.

## Fast startup: config first, weights in the background

The pipeline is ordered so you see a graph almost immediately:

1. **Model Discovery** downloads only `config.json` (small, fast).
2. **Weight Download + Weight Converter** start immediately in a genuine
   background thread -- they're the only slow, multi-GB part, and nothing
   before the final artifact listing actually needs the real weights.
3. **Compatibility** builds the graph via `AutoModel.from_config(config)`
   instead of `AutoModel.from_pretrained()` -- same module tree, same shapes,
   randomly-initialized tensors, **no weight download required for this
   step at all**. This is what actually removes the old slowness, rather
   than just moving it to a background thread.
4. INI Generator, Graph Builder, C++ Generator, Compiler/Auto-Fix, and
   Profiler all run on the main thread while weights continue downloading.
5. The pipeline waits for the background weight download only right before
   **Artifact Manager**, so the final file listing is always complete.

## 30-day workspace cache

Two independent things are cached per model, each expiring after 30 days,
at `<workspace>/.ai_compiler_cache/<model name>/` (sits next to, not inside,
the per-run `nntrainer_out/` folder, so it survives across runs):

- **Compatibility/graph result** (`graph_ir.json`, `report.json`) -- reused
  instantly on the next run of the same model instead of rebuilding the
  module tree.
- **Downloaded weights** and their **converted nntrainer binary** -- reused
  instead of re-downloading a multi-GB checkpoint every run.

Logs clearly say when something came from cache ("Using cached weights (3.2
days old, 1350.4 MB, cache expires after 30 days)") versus being freshly
fetched. After 30 days, the next run re-downloads/re-checks automatically --
there's no manual cache-clearing step needed, though you can delete
`.ai_compiler_cache/` in the workspace at any time to force a full refresh
sooner.

## How the tracer works (no torch.fx)

The **Compatibility Agent** uses a **module-tree walker** instead of `torch.fx.symbolic_trace`:
1. Walks `model.named_modules()` and keeps only **leaf modules** (container
   modules like `BertEncoder`, `BertLayer`, `ModuleList` are skipped -- they
   aren't ops themselves, just wrappers around the real ones)
2. Maps each leaf module's class name (e.g., `Linear`, `LayerNorm`, `Embedding`) to the nntrainer op table
3. Infers input/output shapes via forward hooks during a dummy pass
4. Builds nodes in traversal order, connecting each to the next

This avoids the data-dependent control flow limitation entirely and works with BERT, GPT,
and any HF transformer that has `if` statements, dynamic routing, etc.

## Known limitations (stated plainly, not hidden)

- The op-to-nntrainer-layer table is intentionally conservative: several real
  compute ops (`matmul`, `bmm`, `scaled_dot_product_attention`, ...)
  have no 1:1 nntrainer primitive today and are left as visible TODOs.
- Shape inference is best-effort: it runs a dummy forward pass with random inputs,
  so the inferred shapes may not match your production data. Inspect the node details
  in the UI if needed.
- Profiler per-layer numbers are a proportional estimate from op weight when
  nntrainer doesn't emit a per-layer trace from the smoke-test binary --
  labeled as such in both the event payload and the UI, never presented as a
  measurement it isn't.
- The Auto-Fix Agent only reacts to real compiler stderr, and is capped at 2
  attempts; it will not invent nntrainer APIs to paper over an unsupported op.
