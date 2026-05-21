# NNTrainer Graph Visualizer

A VS Code extension for visualizing, verifying, and profiling the conversion of
PyTorch models to NNTrainer format via TorchFXConverter.

## Features

### Dual Graph Visualization
- **PyTorch FX Graph**: original computational graph with `call_module`, `call_function`, `call_method` nodes
- **NNTrainer Graph**: converted layer graph with `fully_connected`, `rms_norm`, `mha_core` etc.
- **DAG Layout**: automatic topological ordering with SVG edge rendering
- **Pattern Grouping**: Attention/FFN blocks detected and outlined with colored labels

### Interactive Mapping
- **Side-by-Side view** with cross-highlighting on node click
- **Mapping Lines**: dashed SVG lines connecting corresponding FX ↔ NN nodes
- **C++ Source Linking**: click a node → corresponding `createLayer()` line highlighted

### Integrity Verification (4 tabs)
- **Summary**: overall PASS/FAIL status per category
- **Mapping**: unmapped FX nodes and NN layers with coverage percentage
- **Topology**: edge relationship comparison (FX args vs NN input_layers)
- **C++ Cross-check**: `createLayer()` type/count validation

### Node Inspector (3 tabs)
- **Properties**: side-by-side FX vs NN parameter comparison with diff highlighting
- **Shapes**: input/output tensor dimension flow with consistency checks
- **Weights**: HF state_dict key → NN layer weight mapping with transpose indicators

### Profiling
- **Per-layer timing**: hook-based profiling with warmup iterations
- **Heatmap overlay**: green→yellow→red color bars on graph nodes
- **Bottleneck detection**: top 20% layers highlighted with red glow
- **Profile panel**: sorted timing list with clickable navigation

### Export
- **Markdown report**: full verification report with mapping, topology, C++, profile data
- **SVG graph**: NNTrainer graph exported as standalone SVG

## Prerequisites

- **VS Code** >= 1.85.0
- **Python 3.8+** with:
  ```bash
  pip install torch transformers
  ```
- **TorchFXConverter** (included in this repository at `tools/TorchFXConverter/`)

## Installation

### From Source

```bash
cd tools/GraphVisualizer
npm install
npm run compile
```

Then press `F5` in VS Code to launch the Extension Development Host.

### Package as VSIX

```bash
npm install -g @vscode/vsce
cd tools/GraphVisualizer
vsce package
# Install: code --install-extension nntrainer-graph-visualizer-0.0.1.vsix
```

## Usage

### 1. Convert a HuggingFace Model

1. Open Command Palette (`Ctrl+Shift+P`)
2. Run **"NNTrainer: Convert Model"**
3. Enter a model ID (e.g., `Qwen/Qwen3-0.6B`) or local path (e.g., `./test_model`)
4. Wait for conversion to complete
5. The dual graph visualizer opens automatically

```
Command Palette → "NNTrainer: Convert Model" → Enter model ID → View results
```

### 2. Convert a Local PyTorch Model

1. Run **"NNTrainer: Convert Local PyTorch Model (.py)"**
2. Select a `.py` file containing an `nn.Module` class
3. Enter the class name (e.g., `MyTransformer`)
4. Optionally provide input shape JSON
5. The visualizer opens with the source code side-by-side

```
Command Palette → "NNTrainer: Convert Local PyTorch Model" → Select file → Enter class name
```

### 3. Profile a Model

1. First convert a model (step 1 or 2 above)
2. Run **"NNTrainer: Profile Model"**
3. Enter the same model ID
4. Profiling runs 5 iterations with warmup
5. Heatmap overlay appears on the existing graph

```
Command Palette → "NNTrainer: Profile Model" → Enter model ID → View heatmap
```

### 4. Load Existing Results

1. Run **"NNTrainer: Open Graph Visualizer"**
2. Select a `conversion_result.json` file

## Toolbar Guide

| Button | Function |
|--------|----------|
| **Side by Side** | Show both FX and NN graphs (default) |
| **NNTrainer Only** | Show only the converted graph |
| **Torch FX Only** | Show only the original FX graph |
| **Groups** | Toggle Attention/FFN block grouping boxes |
| **Mapping Lines** | Toggle dashed mapping lines between graphs |
| **Profile** | Toggle profile panel (after profiling) |
| **Verification** | Toggle verification report panel |
| **Inspector** | Toggle node property inspector |
| **PyTorch Source** | Toggle PyTorch source code panel |
| **C++ Output** | Toggle generated C++ code panel |
| **Export** | Export verification report (MD) and graph (SVG) |

## Interaction Guide

### Node Selection
- **Click** a node in either graph to select it
- The corresponding node in the other graph is automatically highlighted and scrolled into view
- Connected edges are highlighted in both graphs
- The Inspector panel updates with the selected node's details
- If C++ panel is open, the corresponding `createLayer()` line is highlighted

### Mapping Lines
- Enable "Mapping Lines" to see visual connections between FX and NN nodes
- Lines become bold when their nodes are selected
- Lines update in real-time as you scroll either pane

### Verification
- Click items in the Mapping/Topology tabs to navigate to the corresponding node
- Coverage badge: green (100%), yellow (>80%), red (<80%)

### Profiling Heatmap
- Bottom bar on each node: width = relative time, color = green (fast) to red (slow)
- Hover to see exact timing (e.g., "0.136ms (16.3%)")
- Red glow = bottleneck (top 20% or >10% of total time)
- Click profile panel items to navigate to graph nodes

## Configuration

Open VS Code Settings and search for `nntrainerGraph`:

| Setting | Default | Description |
|---------|---------|-------------|
| `nntrainerGraph.pythonPath` | `python3` | Path to Python with torch/transformers |
| `nntrainerGraph.converterPath` | (auto-detected) | Path to TorchFXConverter directory |
| `nntrainerGraph.defaultSeqLen` | `8` | Sequence length for model tracing |

## Example Workflow

```bash
# 1. Quick test with included tiny model
#    Command Palette → "NNTrainer: Convert Model"
#    Enter: ./tools/TorchFXConverter/test_model

# 2. Profile the same model
#    Command Palette → "NNTrainer: Profile Model"
#    Enter: ./tools/TorchFXConverter/test_model

# 3. Export results
#    Click "Export" button in toolbar
#    Save conversion-report.md and graph.svg
```

## Architecture

See [DESIGN.md](DESIGN.md) for detailed architecture documentation including:
- System architecture diagram
- Class diagram
- Sequence diagrams (conversion, profiling, selection, export flows)
- Webview component design
- Data flow diagram

## Development

```bash
# Install dependencies
cd tools/GraphVisualizer
npm install

# Compile TypeScript
npm run compile

# Watch mode
npm run watch

# Launch Extension Development Host
# Press F5 in VS Code
```

### Project Structure

```
src/
├── extension.ts              # Command registration, entry point
├── converterRunner.ts        # Python bridge, process management
├── graphVisualizerPanel.ts   # Webview panel lifecycle
├── modelExplorerProvider.ts  # Sidebar tree view
├── nodePropertiesProvider.ts # Node detail tree view
└── types.ts                  # TypeScript interfaces

webview/
└── graphView.html            # Complete webview (HTML+CSS+JS)
                              # DAG layout, SVG edges, verification,
                              # profiling, inspector, grouping, export
```
