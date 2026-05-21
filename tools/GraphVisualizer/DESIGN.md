# GraphVisualizer Design Document

## 1. Overview

GraphVisualizer is a VS Code extension that visualizes and verifies the conversion
of PyTorch models to NNTrainer format via TorchFXConverter. It provides:

- **Dual graph rendering** of PyTorch FX graph and NNTrainer layer graph
- **Node-level mapping** between source and target graphs
- **Integrity verification** with topology comparison and C++ cross-checking
- **Per-layer profiling** with heatmap overlay and bottleneck detection
- **Interactive inspection** of properties, shapes, and weight mappings

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VS Code Extension                        │
│                                                                 │
│  ┌──────────┐  ┌────────────────┐  ┌─────────────────────────┐ │
│  │extension  │  │ConverterRunner │  │ GraphVisualizerPanel    │ │
│  │.ts        │──│                │──│                         │ │
│  │(commands) │  │ runConversion()│  │ createOrShow(result)    │ │
│  │           │  │ runProfile()   │  │ sendProfileData(data)   │ │
│  └──────────┘  └───────┬────────┘  └───────────┬─────────────┘ │
│                        │ spawn()                │ webview.html  │
│  ┌──────────────┐      │                        │               │
│  │ModelExplorer  │      │               ┌───────▼─────────────┐ │
│  │Provider       │      │               │  graphView.html      │ │
│  │(tree view)    │      │               │  ┌────────────────┐  │ │
│  └──────────────┘      │               │  │ DAG Layout     │  │ │
│  ┌──────────────┐      │               │  │ SVG Edges      │  │ │
│  │NodeProperties │      │               │  │ Mapping Lines  │  │ │
│  │Provider       │      │               │  │ Heatmap        │  │ │
│  │(tree view)    │      │               │  │ Verification   │  │ │
│  └──────────────┘      │               │  │ Inspector      │  │ │
│                        │               │  │ Groups         │  │ │
│                        │               │  └────────────────┘  │ │
│                        │               └──────────────────────┘ │
└────────────────────────┼────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Python Subprocess   │
              │                      │
              │  vscode_bridge.py    │
              │  (conversion)        │
              │                      │
              │  vscode_profile.py   │
              │  (profiling)         │
              │                      │
              │  ┌────────────────┐  │
              │  │TorchFXConverter│  │
              │  │ converter.py   │  │
              │  │ custom_models  │  │
              │  └────────────────┘  │
              └──────────────────────┘
```

---

## 3. Class Diagram

```
┌─────────────────────────────────────────────┐
│               <<VS Code Extension>>          │
│                  extension.ts                │
│─────────────────────────────────────────────│
│ + activate(context: ExtensionContext): void  │
│ + deactivate(): void                        │
│─────────────────────────────────────────────│
│ Registers commands:                          │
│  - nntrainerGraph.convert                    │
│  - nntrainerGraph.convertLocalModel          │
│  - nntrainerGraph.profile                    │
│  - nntrainerGraph.openVisualizer             │
└──────┬───────────┬───────────┬──────────────┘
       │           │           │
       ▼           ▼           ▼
┌──────────────┐ ┌─────────────────────┐ ┌──────────────────────┐
│ConverterRunner│ │GraphVisualizerPanel │ │ModelExplorerProvider │
│──────────────│ │─────────────────────│ │──────────────────────│
│-context      │ │-panel: WebviewPanel │ │-result               │
│──────────────│ │-extensionUri        │ │──────────────────────│
│+runConversion│ │-disposables         │ │+setConversionResult()│
│ (modelId)    │ │─────────────────────│ │+getTreeItem()        │
│  :Result|null│ │+createOrShow()      │ │+getChildren()        │
│              │ │ (uri, result)       │ └──────────────────────┘
│+runLocal     │ │+sendProfileData()   │
│ Conversion() │ │ (profileData)       │ ┌──────────────────────┐
│  :Result|null│ │+update(result)      │ │NodePropertiesProvider│
│              │ │-getHtml(result)     │ │──────────────────────│
│+runProfile   │ │ :string             │ │+setNode()            │
│ (modelId)    │ │-handleMessage(msg)  │ │+getTreeItem()        │
│  :Profile|nul│ │-saveExport()        │ │+getChildren()        │
│              │ └─────────────────────┘ └──────────────────────┘
│+loadFromJson │
│ (path)       │
│  :Result|null│
│              │
│-getConverter │
│ Path():string│
│-getPythonPath│
│ ():string    │
│-createBridge │
│ Script()     │
└──────────────┘

┌────────────────────────────────────────────────────────┐
│                      types.ts                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ConversionResult ─────────┐                           │
│  │ nntrainerLayers[]       │                           │
│  │ fxGraph[]               │                           │
│  │ modelStructure          │                           │
│  │ weightMap[]             │                           │
│  │ nodeMapping[]           │                           │
│  │ unsupportedOps[]        │     NNTrainerLayer        │
│  │ unknownLayers[]         │     │ name                │
│  │ decomposedModules[]     │     │ layer_type          │
│  │ cppSource               │     │ properties          │
│  │ iniConfig               │     │ input_layers[]      │
│  │ torchSourceCode         │     │ fx_node_name        │
│  │ torchSourcePath         │     │ hf_module_name      │
│  └─────────────────────────┘     │ has_weight/bias     │
│                                  │ weight_hf_key       │
│  FxNode ───────────────────┐     │ transpose_weight    │
│  │ name                    │     └─────────────────────│
│  │ op (placeholder|        │                           │
│  │     call_module|...)    │     NodeMapping            │
│  │ target                  │     │ fxNodeName           │
│  │ args[]                  │     │ nntrainerLayerName   │
│  │ output_shape            │     │ mappingType          │
│  │ module_type             │     └─────────────────────│
│  │ scope                   │                           │
│  │ meta                    │     ProfileData            │
│  └─────────────────────────┘     │ total_time_ms       │
│                                  │ layers: LayerProfile│
│  ModelStructure                  │ bottlenecks[]       │
│  │ model_type, arch_type   │     └─────────────────────│
│  │ blocks: TransformerBlock│                           │
│  │   ├─ AttentionPattern   │     LayerProfile          │
│  │   └─ FFNPattern         │     │ name, layer_type   │
│  └─────────────────────────┘     │ time_ms, memory_mb │
│                                  │ flops, pct_of_total│
│                                  └─────────────────────│
└────────────────────────────────────────────────────────┘
```

---

## 4. Sequence Diagrams

### 4.1 Model Conversion Flow

```
User          Extension        ConverterRunner      Python(bridge)     Webview
 │                │                  │                    │               │
 │ Convert cmd    │                  │                    │               │
 │───────────────>│                  │                    │               │
 │                │ showInputBox()   │                    │               │
 │ "Qwen/Qwen3"  │                  │                    │               │
 │<───────────────│                  │                    │               │
 │                │ runConversion()  │                    │               │
 │                │─────────────────>│                    │               │
 │                │                  │ spawn(python,      │               │
 │                │                  │  vscode_bridge.py) │               │
 │                │                  │───────────────────>│               │
 │                │                  │  PROGRESS: Loading │               │
 │                │  progress.report │<──────────────────│               │
 │                │<─────────────────│                    │               │
 │                │                  │  PROGRESS: Running │               │
 │                │  progress.report │<──────────────────│               │
 │                │<─────────────────│                    │               │
 │                │                  │  conversion_result │               │
 │                │                  │  .json written     │               │
 │                │                  │<──────────────────│               │
 │                │  ConversionResult│                    │               │
 │                │<─────────────────│                    │               │
 │                │ createOrShow(result)                  │               │
 │                │──────────────────────────────────────>│               │
 │                │                  │                    │  getHtml()    │
 │                │                  │                    │  template +   │
 │                │                  │                    │  JSON.stringify│
 │                │                  │                    │──────────────>│
 │                │                  │                    │  renderFx()   │
 │                │                  │                    │  renderNn()   │
 │                │                  │                    │  renderVer()  │
 │                │                  │                    │  drawGroups() │
 │                │                  │                    │               │
 │  Visualizer    │                  │                    │               │
 │  shown         │                  │                    │               │
 │<───────────────│                  │                    │               │
```

### 4.2 Profiling Flow

```
User          Extension        ConverterRunner      Python(profile)    Webview
 │                │                  │                    │               │
 │ Profile cmd    │                  │                    │               │
 │───────────────>│                  │                    │               │
 │                │ runProfile()     │                    │               │
 │                │─────────────────>│                    │               │
 │                │                  │ spawn(python,      │               │
 │                │                  │  vscode_profile.py)│               │
 │                │                  │───────────────────>│               │
 │                │                  │  PROGRESS: Loading │               │
 │                │                  │<──────────────────│               │
 │                │                  │  PROGRESS: Running │               │
 │                │                  │  (N iterations)    │               │
 │                │                  │<──────────────────│               │
 │                │                  │  profile_result    │               │
 │                │                  │  .json             │               │
 │                │  ProfileData     │<──────────────────│               │
 │                │<─────────────────│                    │               │
 │                │ sendProfileData()│                    │               │
 │                │─────────────────────────────────────>│               │
 │                │                  │                    │  postMessage  │
 │                │                  │                    │──────────────>│
 │                │                  │                    │  applyProfile │
 │                │                  │                    │  Heatmap()    │
 │                │                  │                    │  renderProfile│
 │                │                  │                    │  Panel()      │
 │  Heatmap       │                  │                    │               │
 │  overlaid      │                  │                    │               │
 │<───────────────│                  │                    │               │
```

### 4.3 Node Selection & Cross-Highlight Flow

```
Webview(FX pane)       Webview(Core)        Webview(NN pane)     Webview(Inspector)
     │                      │                     │                    │
     │ click node           │                     │                    │
     │─────────────────────>│                     │                    │
     │                      │ pickFx(name)        │                    │
     │                      │ ── clearSel()       │                    │
     │                      │ ── add .sel class   │                    │
     │  FX edges highlight  │ ── hlEdges('eg-fx') │                    │
     │<─────────────────────│                     │                    │
     │                      │ ── lookup mapping   │                    │
     │                      │    mByFx.get(name)  │                    │
     │                      │                     │                    │
     │                      │ ── add .sel + .hl   │                    │
     │                      │ ── scrollIntoView   │                    │
     │                      │────────────────────>│                    │
     │                      │ ── hlEdges('eg-nn') │                    │
     │                      │────────────────────>│                    │
     │                      │ ── drawMap()        │                    │
     │                      │ (mapping lines SVG) │                    │
     │                      │                     │                    │
     │                      │ ── renderInspector() │                   │
     │                      │─────────────────────────────────────────>│
     │                      │                     │  Properties tab    │
     │                      │                     │  Shapes tab        │
     │                      │                     │  Weights tab       │
     │                      │                     │                    │
     │                      │ ── highlightCppLine()│                   │
     │                      │ (C++ panel highlight)│                   │
     │                      │                     │                    │
     │                      │ ── postMessage      │                    │
     │                      │   (nodeSelected)    │                    │
     │                      │───> Extension       │                    │
```

### 4.4 Export Flow

```
User          Webview              Extension             FileSystem
 │               │                      │                    │
 │ Click Export  │                      │                    │
 │──────────────>│                      │                    │
 │               │ exportReport()       │                    │
 │               │ ── generate MD       │                    │
 │               │ ── generate SVG      │                    │
 │               │                      │                    │
 │               │ postMessage          │                    │
 │               │ (exportReport, {md}) │                    │
 │               │─────────────────────>│                    │
 │               │                      │ showSaveDialog()   │
 │               │                      │───────────────────>│
 │               │                      │ writeFileSync()    │
 │               │                      │───────────────────>│
 │               │ postMessage          │                    │
 │               │ (exportSvg, {svg})   │                    │
 │               │─────────────────────>│                    │
 │               │                      │ showSaveDialog()   │
 │               │                      │───────────────────>│
 │  "Exported"   │                      │                    │
 │<──────────────│                      │                    │
```

---

## 5. Webview Component Design

```
┌─ graphView.html ──────────────────────────────────────────────────┐
│                                                                    │
│ ┌─ Toolbar ──────────────────────────────────────────────────────┐ │
│ │ [SideBySide] [NN Only] [FX Only] │ [Groups] [Mapping] [Prof]  │ │
│ │ [Verify] [Inspector] │ [Src] [C++] [Export] │ Stats            │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│ ┌─ Main Container ───────────────────────────────────────────────┐ │
│ │                                                                 │ │
│ │ ┌─PyTorch Src─┐ ┌─FX Graph────────┐ ┌─NN Graph────────┐       │ │
│ │ │ (hidden)    │ │ SVG edges       │ │ SVG edges       │       │ │
│ │ │ line nums   │ │ ┌─┐ ┌─┐ ┌─┐    │ │ ┌─┐ ┌─┐ ┌─┐    │       │ │
│ │ │ highlight   │ │ │ │→│ │→│ │    │ │ │ │→│ │→│ │    │       │ │
│ │ │             │ │ └─┘ └─┘ └─┘    │ │ └─┘ └─┘ └─┘    │       │ │
│ │ │             │ │   ┌─┐   ┌─┐    │ │   ┌─┐   ┌─┐    │       │ │
│ │ │             │ │   │ │→→→│ │    │ │   │ │→→→│ │    │       │ │
│ │ │             │ │   └─┘   └─┘    │ │   └─┘   └─┘    │       │ │
│ │ │             │ │ ╭──Attn L0──╮  │ │ ╭──Attn L0──╮  │       │ │
│ │ │             │ │ │ q_proj    │  │ │ │ q_proj    │  │ ┌───┐ │ │
│ │ │             │ │ │ k_proj    │  │ │ │ k_proj    │  │ │   │ │ │
│ │ │             │ │ │ v_proj    │  │ │ │ v_proj    │  │ │ V │ │ │
│ │ │             │ │ ╰───────────╯  │ │ ╰───────────╯  │ │ e │ │ │
│ │ │             │ │ Profile bars   │ │ Profile bars   │ │ r │ │ │
│ │ │             │ │ ██████ 15.2%  │ │ ██████ 15.2%  │ │ i │ │ │
│ │ │             │ │ ███ 7.1%     │ │ ███ 7.1%     │ │ f │ │ │
│ │ └─────────────┘ └────────────────┘ └────────────────┘ │ y │ │ │
│ │                                                        │   │ │ │
│ │               ┌─SVG Mapping Overlay──────────────┐     │   │ │ │
│ │               │  ╌╌╌╌╌  dashed lines  ╌╌╌╌╌╌╌  │     │   │ │ │
│ │               │  connecting FX ↔ NN nodes        │     │   │ │ │
│ │               └──────────────────────────────────┘     │   │ │ │
│ │                                                        └───┘ │ │
│ │ ┌─C++ Src─────┐ ┌─Profile Panel──┐ ┌─Inspector──────────────┐│ │
│ │ │ (hidden)    │ │ Total: 1.86ms  │ │ [Props][Shapes][Weight] ││ │
│ │ │ createLayer │ │ Bottlenecks:   │ │ FX: model_embed_tokens  ││ │
│ │ │ >>highlight │ │  █ rotary 16%  │ │ NN: model_embed_tokens  ││ │
│ │ │             │ │  █ q_norm 7%   │ │                         ││ │
│ │ │             │ │ All Layers:    │ │ num_embeddings: 1000    ││ │
│ │ │             │ │  rotary 0.14ms │ │ in_dim: 1000  ✓ match  ││ │
│ │ │             │ │  q_norm 0.06ms │ │                         ││ │
│ │ └─────────────┘ └────────────────┘ │ Weight: TRANSPOSE       ││ │
│ │                                    │ model.embed.weight →    ││ │
│ │                                    │ model_embed.weight      ││ │
│ │                                    └─────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 5.1 JS Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `dagLayout()` | Longest-path DAG layering algorithm for node positioning |
| `renderFx()` | FX graph nodes + SVG edges with op-type color coding |
| `renderNn()` | NNTrainer graph nodes + SVG edges with layer-type colors |
| `drawMap()` | Cross-panel SVG mapping lines (requestAnimationFrame) |
| `drawGroups()` | Detect Attention/FFN blocks by name pattern, draw group boxes |
| `pickFx()/pickNn()` | Node selection with cross-highlight, C++ linking, inspector update |
| `verifyMapping()` | Node mapping coverage analysis |
| `verifyTopology()` | FX edge vs NN input_layers comparison |
| `verifyCpp()` | createLayer() type matching against NN layers |
| `renderInspector()` | 3-tab property/shape/weight inspector |
| `applyProfileHeatmap()` | Apply per-node color bars from profile data |
| `exportReport()` | Generate Markdown report + SVG graph |

---

## 6. Data Flow

```
┌──────────────────────┐     ┌────────────────────────────────────┐
│ HuggingFace Model    │     │ Local .py Model                    │
│ (model ID or path)   │     │ (nn.Module class)                  │
└──────────┬───────────┘     └──────────────┬─────────────────────┘
           │                                │
           ▼                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    vscode_bridge.py                               │
│  1. Load model (from_pretrained / from_config / dynamic import)  │
│  2. torch.fx.symbolic_trace() via AdaptiveConverter              │
│  3. Serialize: FX graph, NN layers, structure, weights, C++/INI  │
│  4. Build node mapping (fxNode ↔ nntrainerLayer)                 │
│  Output: conversion_result.json                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ConversionResult (JSON)                         │
│  ┌─────────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐  │
│  │nntrainerLay.│ │ fxGraph  │ │ nodeMapping │ │ weightMap   │  │
│  │(100 layers) │ │(122 node)│ │(120 maps)   │ │(25 entries) │  │
│  └─────────────┘ └──────────┘ └─────────────┘ └──────────────┘  │
│  ┌─────────────┐ ┌──────────┐ ┌─────────────┐                   │
│  │modelStructur│ │ cppSource│ │ iniConfig   │                   │
│  │(2 blocks)   │ │(8.3 KB)  │ │(3.6 KB)     │                   │
│  └─────────────┘ └──────────┘ └─────────────┘                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   graphView.html (Webview)                        │
│                                                                  │
│  Data → dagLayout() → renderFx() + renderNn()                   │
│       → verifyMapping() + verifyTopology() + verifyCpp()         │
│       → renderInspector() + drawGroups()                         │
│                                                                  │
│  ProfileData (via postMessage) → applyProfileHeatmap()           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. File Structure

```
tools/GraphVisualizer/
├── package.json                 # Extension manifest, commands, config
├── tsconfig.json                # TypeScript configuration
├── src/
│   ├── extension.ts             # Entry point, command registration (134 lines)
│   ├── converterRunner.ts       # Python process spawning, bridge script (890 lines)
│   ├── graphVisualizerPanel.ts  # Webview panel management (145 lines)
│   ├── modelExplorerProvider.ts # Tree view for model structure (137 lines)
│   ├── nodePropertiesProvider.ts# Tree view for selected node props (30 lines)
│   └── types.ts                 # TypeScript interfaces (143 lines)
├── webview/
│   └── graphView.html           # Full webview UI (1622 lines)
├── resources/
│   └── icon.svg                 # Activity bar icon
└── out/                         # Compiled JS output

tools/TorchFXConverter/
├── vscode_bridge.py             # Conversion bridge for VS Code
└── vscode_profile.py            # Profiling bridge for VS Code
```

---

## 8. Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `nntrainerGraph.pythonPath` | `python3` | Python interpreter path |
| `nntrainerGraph.converterPath` | (auto) | TorchFXConverter directory |
| `nntrainerGraph.defaultSeqLen` | `8` | Default sequence length for tracing |

---

## 9. Extension Commands

| Command | Title | Description |
|---------|-------|-------------|
| `nntrainerGraph.convert` | NNTrainer: Convert Model | Convert HuggingFace model |
| `nntrainerGraph.convertLocalModel` | NNTrainer: Convert Local PyTorch Model | Convert local .py file |
| `nntrainerGraph.profile` | NNTrainer: Profile Model | Profile model execution |
| `nntrainerGraph.openVisualizer` | NNTrainer: Open Graph Visualizer | Load existing JSON result |
