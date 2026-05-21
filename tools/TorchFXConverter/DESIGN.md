# Torch.FX → NNTrainer Symbolic Tensor Graph Converter

## Design Philosophy

This converter is **architecture-agnostic**. Rather than hardcoding patterns for
specific models (Qwen3, BERT, T5, etc.), it:

1. **Traces** any HuggingFace model using callback-based FX tracing
2. **Maps** traced leaf modules to NNTrainer layer types via a registry
3. **Detects patterns** (attention, FFN, residual connections) from graph topology
4. **Emits** NNTrainer **symbolic tensor graph** C++ code that uses `Tensor` +
   `LayerHandle` to build the model declaratively

The generated C++ code follows the same **symbolic tensor graph** pattern used
across all NNTrainer applications (ResNet, MNIST, YOLOv2/v3, CausalLM, etc.).

### Supported Architectures
- **Decoder-only**: Qwen3, LLaMA, GPT-2, Mistral
- **Encoder-only**: BERT, RoBERTa, sentence-transformers
- **Encoder-Decoder**: T5, mT5, BART

## NNTrainer Symbolic Tensor Graph

All NNTrainer applications have been converted to a **symbolic tensor graph**
construction pattern. Instead of manually specifying `input_layers` connections
between layers, models are built by chaining symbolic `Tensor` objects through
`LayerHandle` calls. The `model->compile(input, output)` method then walks the
graph backward to discover all layers and their connections automatically.

### Core API

```cpp
// 1. Create symbolic input tensor (no data, just a placeholder)
Tensor x = Tensor({1, 1, 28, 28}, "inputlayer");

// 2. Create layers via LayerHandle (wraps std::shared_ptr<Layer>)
LayerHandle conv(createLayer("conv2d", {"filters=32", "kernel_size=3,3"}));
LayerHandle fc(createLayer("fully_connected", {"unit=10"}));

// 3. Chain layers: each call creates a new symbolic tensor
Tensor h = conv(x);        // conv takes x, produces h
Tensor y = fc(h);           // fc takes h, produces y

// 4. Multi-input layers (addition, concat, etc.)
LayerHandle add(createLayer("Addition", {"name=res_add"}));
Tensor merged = add({branch_a, branch_b});

// 5. Compile: walks tensor graph backward from output to input
model->compile(x, y, ml::train::ExecutionMode::INFERENCE);

// 6. Multi-input/output overloads
model->compile(input, outputs_vec, mode);          // single-in, multi-out
model->compile(inputs_vec, outputs_vec, mode);     // multi-in, multi-out
```

### Tensor API Operations

The `Tensor` class also supports symbolic operations that implicitly create
layers in the graph:

```cpp
Tensor result = a.add(b);           // Addition layer
Tensor result = a.multiply(b);      // Multiply layer
Tensor result = a.dot(b);           // MatMul layer
Tensor result = a.reshape(dim);     // Reshape layer
Tensor result = a.sum(axis);        // Sum along axis
Tensor result = a.average(axis);    // Average along axis
```

### Graph Navigation

```cpp
std::shared_ptr<Layer> layer = tensor.getProducingLayer();
std::vector<Tensor> inputs = tensor.getInputTensors();
Tensor nth_output = multi_out_tensor.output(n);
```

### Compile Pipeline

```
User builds Tensor graph via LayerHandle calls
        │
        ▼
model->compile(input_tensor, output_tensor)
        │
        ├── Traverses graph backward from output via getProducingLayer()
        ├── Collects all layers and builds input_layers connections
        ├── Adds layers to model in topological order
        ├── Calls internal compile() to initialize shapes and allocate memory
        └── Model ready for training / inference
```

## Converter Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Tracer     │───►│  Node Mapper │───►│   Pattern    │───►│   Emitter    │
│  (FX Graph)  │    │  (Leaf→NNTR) │    │  Detector    │    │  (C++/JSON)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                    │                   │                    │
   Callback-based     Module type →        Graph topology      Symbolic tensor
   hook tracing       NNTrainer layer      analysis             graph C++ code
```

### Module Layout

```
tools/TorchFXConverter/
├── tracer.py              # FX graph tracer (callback-based)
├── node_mapper.py         # Maps FX nodes → NNTrainer layer definitions
├── decomposer.py          # Adaptive multi-pass conversion pipeline
├── pattern_detector.py    # Detects attention/FFN/residual patterns
├── emitter_cpp/           # Generates C++ symbolic tensor graph code
│   ├── header.py          #   Header file generation
│   ├── helpers.py         #   Utility functions
│   ├── source_construct.py#   constructModel() with Tensor graph
│   ├── source_block.py    #   Transformer block methods
│   ├── source_attention.py#   Attention block methods
│   ├── source_ffn.py      #   FFN block methods
│   └── source_custom.py   #   Custom layer support
├── emitter_ini/           # Generates NNTrainer .ini configuration
│   ├── flat.py            #   Flat model INI output
│   ├── structured.py      #   Structured block INI output
│   └── helpers.py         #   INI formatting utilities
├── emitter_json.py        # Generates JSON model config + weight map
├── weight_converter.py    # HF → NNTrainer binary weight conversion
├── converter.py           # Main CLI entry point
├── nntrainer_layers.py    # NNTrainer layer type definitions & registry
├── op_registry.py         # Operation lookup tables
├── module_mapper.py       # nn.Module → NNTrainer layer mapping
├── function_mapper.py     # torch.* function mapping
├── method_mapper.py       # Tensor.* method mapping
├── mapper_helpers.py      # Shared mapper utilities
├── plugin_registry.py     # Custom layer plugin registration system
├── plugin_codegen.py      # C++ LayerPluggable boilerplate generator
├── patterns/              # Pattern detection modules
├── DESIGN.md              # This file
├── ARCHITECTURE.md        # Detailed architecture with mermaid diagrams
└── tests/
    ├── test_tracer_simple.py
    ├── test_tracer_qwen3.py
    ├── test_node_mapper.py
    ├── test_decomposer.py
    ├── test_pattern_detector.py
    ├── test_pattern_modules.py
    ├── test_coverage.py
    ├── test_multi_arch.py
    ├── test_unmapped_ops.py
    ├── test_emitters.py
    ├── test_emitter_cpp_modules.py
    ├── test_emitter_ini_modules.py
    ├── test_new_layer_mappers.py
    ├── test_tier2_layer_mappers.py
    ├── test_tier3_layer_mappers.py
    ├── test_mapper_helpers.py
    ├── test_plugin_system.py
    ├── test_e2e.py
    └── test_build_and_run.py
```

## Generated Code Examples

### Simple Model (MNIST-style)

The converter generates C++ code that mirrors the hand-written application
pattern:

```cpp
void Model::constructModel() {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({"loss=cross"});

  LayerHandle input_layer(createLayer("input", {
    withKey("name", "input0"),
    withKey("input_shape", "1:28:28")
  }));
  Tensor input = input_layer(Tensor());

  LayerHandle conv1(createLayer("conv2d", {
    withKey("name", "conv1"), withKey("filters", 6),
    withKey("kernel_size", "5,5")
  }));
  LayerHandle pool1(createLayer("pooling2d", {
    withKey("name", "pool1"), withKey("pool_size", "2,2"),
    withKey("pooling", "average")
  }));
  LayerHandle fc(createLayer("fully_connected", {
    withKey("name", "fc1"), withKey("unit", 10),
    withKey("activation", "softmax")
  }));

  Tensor h = conv1(input);
  h = pool1(h);
  Tensor output = fc(h);

  model->compile(input, output, ml::train::ExecutionMode::INFERENCE);
}
```

### Transformer Model (CausalLM-style)

For transformer models, the converter generates structured code with block
construction methods:

```cpp
void Model::constructModel() {
  // Input layer
  LayerHandle input_layer(createLayer("input", {
    withKey("name", "input0"),
    withKey("input_shape", "1:1:" + std::to_string(SEQ_LEN))
  }));
  Tensor input = input_layer(Tensor());

  // Embedding
  LayerHandle embedding(createLayer("embedding_layer", {
    withKey("name", "embedding0"),
    withKey("in_dim", NUM_VOCAB), withKey("out_dim", DIM)
  }));
  Tensor x = embedding(input);

  // External KV cache inputs (for autoregressive inference)
  std::vector<Tensor> all_inputs = {input};
  for (int i = 0; i < NUM_LAYERS; ++i) {
    // ... create key/value cache input tensors
    all_inputs.push_back(key_cache_tensors[i]);
    all_inputs.push_back(val_cache_tensors[i]);
  }

  // Transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    x = createTransformerDecoderBlock(i, x);
  }

  // Final norm + LM head
  LayerHandle output_norm(createLayer("rms_norm", {...}));
  LayerHandle lmhead(createLayer("fully_connected", {...}));
  x = output_norm(x);
  Tensor output = lmhead(x);

  // Multi-input compile (token input + KV cache inputs)
  std::vector<Tensor> outputs = {output};
  model->compile(all_inputs, outputs, ml::train::ExecutionMode::INFERENCE);
}
```

### Branching Topology (YOLOv2-style)

The symbolic graph naturally handles branches, merges, and skip connections:

```cpp
// Main backbone
auto h = convBlock("backbone", input, 64, 3, true);

// Branch A: downsample path
auto branch_a = pool(h);
branch_a = convBlock("branch_a", branch_a, 256, 3, false);

// Branch B: feature extraction path
auto branch_b = convBlock("branch_b", h, 64, 1, false);
branch_b = reorg(branch_b);

// Merge branches via concat
LayerHandle concat_layer(createLayer("concat", {"name=merge"}));
h = concat_layer({branch_a, branch_b});

// ResNet-style skip connection via addition
LayerHandle add_layer(createLayer("Addition", {"name=res_add"}));
h = add_layer({main_path, skip_path});
```

## NNTrainer Layer Type Mapping

| HuggingFace Module          | NNTrainer Layer Type     | Key Properties                          |
|-----------------------------|--------------------------|----------------------------------------|
| nn.Linear                   | fully_connected          | unit, disable_bias, weight_dtype       |
| nn.Embedding                | embedding_layer          | in_dim, out_dim, weight_dtype          |
| nn.Embedding (tied)         | tie_word_embeddings      | in_dim, out_dim, shared_from           |
| *RMSNorm                    | rms_norm                 | epsilon, packed                        |
| *RMSNorm (reshaped, Q/K)    | reshaped_rms_norm        | epsilon, feature_size                  |
| nn.LayerNorm                | layer_norm               | (for BERT/T5)                          |
| nn.Conv2d                   | conv2d                   | filters, kernel_size, stride, padding  |
| nn.ConvTranspose2d          | conv2dtranspose          | filters, kernel_size, stride, padding  |
| nn.Conv2d (depthwise)       | depthwiseconv2d          | filters, kernel_size, stride, padding  |
| nn.GroupNorm                 | group_normalization      | num_groups, epsilon                    |
| nn.InstanceNorm1d/2d        | instance_normalization   | epsilon                                |
| nn.MultiheadAttention       | multi_head_attention     | num_heads, projected_key_dim           |
| nn.GRUCell/LSTMCell/RNNCell | grucell/lstmcell/rnncell | unit                                   |
| nn.Identity                 | identity                 | —                                      |
| nn.CrossEntropyLoss         | cross_softmax            | —                                      |
| nn.MSELoss                  | mse                      | —                                      |
| nn.KLDivLoss                | kld                      | —                                      |
| nn.BCEWithLogitsLoss        | cross_sigmoid            | —                                      |
| Attention pattern           | mha_core                 | num_heads, num_heads_kv, rope_theta... |
| SwiGLU pattern              | swiglu                   | input_layers                           |
| Residual add                | addition                 | input_layers                           |
| Input placeholder           | input                    | input_shape                            |
| Custom (via plugin)         | user-defined             | via PluginRegistry                     |

## Pattern Detection Strategy

Instead of matching fixed subgraph patterns (which breaks on new architectures),
we detect patterns from **module hierarchy** and **data flow**:

### 1. Attention Detection
- Find groups of Linear layers whose outputs flow into a common "attention" operation
- Identify Q/K/V/O projections by:
  - Module name matching (q_proj, k_proj, v_proj, o_proj / query, key, value)
  - Shape analysis (Q has num_heads * head_dim, K/V have num_kv_heads * head_dim)
- Detect post-projection norms (Qwen3's reshaped_rms_norm on Q/K)

### 2. FFN Detection
- Find groups of Linear layers within MLP/FFN modules
- Detect gate mechanism:
  - SwiGLU: up_proj + gate_proj → SiLU(gate) * up → down_proj
  - GELU FFN: fc1 → GELU → fc2 (BERT-style)
  - Standard FFN: fc1 → ReLU → fc2

### 3. Residual Connection Detection
- Track tensor flow: if output = input + sublayer_output, emit `addition` layer
- Detected from `torch.add` / `operator.add` in the FX graph

### 4. Normalization Detection
- RMSNorm modules → rms_norm
- LayerNorm modules → layer_norm (BERT/T5)
- Position: pre-norm (LLaMA/Qwen3) vs post-norm (BERT original)

## Applications Converted to Symbolic Tensor Graph

All NNTrainer applications now use the symbolic tensor graph pattern:

| Application      | Topology              | Key Features                          |
|------------------|-----------------------|---------------------------------------|
| SimpleFC         | Linear chain          | 56-layer FC stack                     |
| MNIST            | CNN pipeline          | Conv → Pool → FC                     |
| ResNet           | Skip connections      | Residual blocks with Addition layers  |
| LLaMA            | Decoder-only          | Transformer blocks, RoPE, GQA        |
| CausalLM         | Decoder-only          | External KV cache, multi-input compile|
| Qwen3            | Decoder-only          | CausalLM subclass, tied embeddings   |
| YOLOv2           | Branching + merge     | Concat of two feature branches       |
| YOLOv3           | Multi-output          | Three detection heads, multi-out compile |
| MixedPrecision   | Linear chain          | Weight dtype properties               |
| ProductRatings   | Multi-input           | Multiple feature inputs               |
| PicoGPT          | Decoder-only          | Minimal GPT, LayerHandle graph        |

## Phase Plan

### Phase 1: Tracer Foundation ✓
- Task 1.1: Setup tracer.py with NNTrainer-adapted LEAF_MODULES ✓
- Task 1.2: Trace a simple model and verify graph ✓
- Task 1.3: Trace Qwen3-0.6B (or tiny config) and dump graph ✓

### Phase 2: NNTrainer Layer Definitions & Node Mapper ✓
- Task 2.1: Create nntrainer_layers.py with NNTrainerLayerDef dataclass ✓
- Task 2.2: Create node_mapper.py mapping leaf modules to NNTrainer types ✓
- Task 2.3: Map function/method nodes (add, matmul, reshape, etc.) ✓
- Task 2.4: Map activation functions (SiLU, GELU, ReLU, softmax) ✓

### Phase 3: Pattern Detection (Architecture-Agnostic) ✓
- Task 3.1: Architecture detector from model.config ✓
- Task 3.2: Attention pattern detection (works for GQA, MHA, MQA) ✓
- Task 3.3: FFN pattern detection (SwiGLU, GELU-FFN, standard FFN) ✓
- Task 3.4: Residual connection & normalization detection ✓
- Task 3.5: Full decoder block assembly ✓
- Task 3.6: Full encoder block assembly (for BERT/T5-encoder) ✓
- Task 3.7: Encoder-decoder assembly (for T5/BART) ✓

### Phase 4: Emitter ✓
- Task 4.1: C++ code emitter → **symbolic tensor graph** (emitter_cpp/) ✓
- Task 4.2: INI config emitter (emitter_ini/) ✓
- Task 4.3: JSON config emitter (emitter_json.py) ✓
- Task 4.4: Weight converter (weight_converter.py) ✓

### Phase 5: End-to-End Validation ✓
- Task 5.1: Qwen3 full pipeline + reference comparison ✓
- Task 5.2: BERT encoder-only test ✓
- Task 5.3: mT5 encoder-decoder test ✓
- Task 5.4: Layer-by-layer comparison with CausalLM reference ✓
- Task 5.5: INI section validation + connectivity check ✓
- Task 5.6: JSON schema validation + roundtrip ✓
- Task 5.7: Weight binary roundtrip verification ✓

### Phase 6: CLI ✓
- Task 6.1: converter.py main entry point ✓
- Task 6.2: Multi-format output (--format cpp ini json) ✓
- Task 6.3: Weight conversion (--weights --dtype float16) ✓

### Phase 7: NNTrainer Application Migration ✓
- Task 7.1: Convert SimpleFC, MNIST, ResNet to symbolic tensor graph ✓
- Task 7.2: Convert LLaMA, CausalLM, Qwen3 to symbolic tensor graph ✓
- Task 7.3: Convert YOLOv2/v3 with multi-output compile support ✓
- Task 7.4: Convert MixedPrecision, ProductRatings, Multi_input ✓
- Task 7.5: Convert TorchFXConverter C++ emitter to emit symbolic graph ✓
- Task 7.6: Add compile(vector<Tensor>, vector<Tensor>) overload ✓
- Task 7.7: Fix graph construction (O(N!) memory fix, input dedup) ✓

## Weight Conversion Rules

NNTrainer expects weights in a specific order:
1. **Linear (fully_connected)**: Transpose weight [out,in] → [in,out]
2. **Embedding**: Keep as-is [vocab, dim]
3. **RMSNorm**: Single weight vector
4. **LayerNorm**: gamma, beta (two vectors)
5. **Tied embeddings**: Shared weight reference

Weight save order follows the layer creation order in constructModel().
