# Gradient Checkpointing

Gradient checkpointing is a memory optimization technique that trades computation for memory by recomputing intermediate activations during the backward pass instead of storing them.

## Overview

When training deep neural networks, intermediate activations from the forward pass must be stored for use during backpropagation. This can consume significant memory, especially for large models. Gradient checkpointing reduces memory usage by:

1. **Discarding** intermediate activations after the forward pass
2. **Recomputing** them during the backward pass when needed

This trades ~20-30% additional computation time for up to **45% memory reduction**.

## Usage

### C++ API

```cpp
#include <model.h>

// Create and configure your model
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

// Add layers
model->addLayer(ml::train::createLayer("input", {"name=input", "input_shape=1:1:784"}));
model->addLayer(ml::train::createLayer("fully_connected", {"name=fc1", "unit=512"}));
model->addLayer(ml::train::createLayer("activation", {"name=act1", "activation=relu"}));
model->addLayer(ml::train::createLayer("fully_connected", {"name=fc2", "unit=256"}));
model->addLayer(ml::train::createLayer("activation", {"name=act2", "activation=relu"}));
model->addLayer(ml::train::createLayer("fully_connected", {"name=fc3", "unit=128"}));
model->addLayer(ml::train::createLayer("activation", {"name=act3", "activation=relu"}));
model->addLayer(ml::train::createLayer("fully_connected", {"name=output", "unit=10"}));
model->addLayer(ml::train::createLayer("mse", {"name=loss"}));

// Define checkpoint blocks BEFORE model initialization
// Each checkpoint block specifies a range of layers whose activations will be recomputed
model->addCheckpointBlock({"fc1", "act1", "fc2", "act2"});  // Checkpoint layers fc1 through act2

// Initialize and compile the model
model->setProperty({"batch_size=32", "epochs=10"});
model->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
model->compile();
model->initialize();

// Train as usual
model->train(dataset);
```

### Multiple Checkpoint Blocks

You can define multiple checkpoint blocks for different sections of your model:

```cpp
// For a transformer-like model with multiple layers
model->addCheckpointBlock({"layer1_attn", "layer1_ffn"});
model->addCheckpointBlock({"layer2_attn", "layer2_ffn"});
model->addCheckpointBlock({"layer3_attn", "layer3_ffn"});
```

## API Reference

### `Model::addCheckpointBlock`

```cpp
int addCheckpointBlock(const std::vector<std::string> &layer_names);
```

**Parameters:**
- `layer_names`: Vector of layer names that form a checkpoint block. Layers must be consecutive in the model graph.

**Returns:**
- `ML_ERROR_NONE` on success
- `ML_ERROR_INVALID_PARAMETER` if layers are not consecutive or model is already initialized

**Notes:**
- Must be called **before** `model->initialize()`
- Layers in a checkpoint block must be consecutive in the forward pass order
- The first layer's input and last layer's output are preserved; intermediate activations are recomputed

## Performance Characteristics

### Memory Savings

| Model Configuration | Memory Reduction |
|---------------------|------------------|
| 2 layers            | ~17-22%          |
| 4 layers            | ~28-39%          |
| 6+ layers           | ~32-46%          |

### Computation Overhead

- Forward pass: Minimal overhead
- Backward pass: ~20-30% additional time (due to recomputation)

### When to Use

**Recommended:**
- Large models that don't fit in memory
- Models with 2+ consecutive layers that can be checkpointed
- Training with large batch sizes

**Not Recommended:**
- Small models where memory is not a concern
- Single-layer checkpoint blocks (overhead exceeds savings)
- Inference-only scenarios (no backward pass)

## Implementation Details

### How It Works

1. **Forward Pass**: 
   - First layer input is saved with `FORWARD_GRAD_LIFESPAN | RECOMPUTE`
   - Intermediate activations use `CALC_GRAD_DERIV_LIFESPAN | RECOMPUTE` (discarded after forward)
   - Last layer output is saved with `FORWARD_GRAD_LIFESPAN | RECOMPUTE`

2. **Backward Pass**:
   - When gradients reach a checkpoint block, the forward pass is re-executed for that block
   - Recomputed activations are used for gradient calculation
   - Activations are discarded again after use

### Execution Order

The implementation extends the execution order tuple from 4 to 5 elements:
```
(forward_order, recompute_order, calc_gradient_order, calc_derivative_order, apply_gradient_order)
```

## Example: Transformer with Checkpointing

```cpp
// Create a transformer model with gradient checkpointing
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

// Add transformer layers
for (int i = 0; i < num_layers; i++) {
    std::string prefix = "layer" + std::to_string(i);
    
    model->addLayer(createLayer("multi_head_attention", 
        {"name=" + prefix + "_attn", "num_heads=8"}));
    model->addLayer(createLayer("layer_normalization", 
        {"name=" + prefix + "_ln1"}));
    model->addLayer(createLayer("fully_connected", 
        {"name=" + prefix + "_ffn1", "unit=2048"}));
    model->addLayer(createLayer("activation", 
        {"name=" + prefix + "_act", "activation=gelu"}));
    model->addLayer(createLayer("fully_connected", 
        {"name=" + prefix + "_ffn2", "unit=512"}));
    model->addLayer(createLayer("layer_normalization", 
        {"name=" + prefix + "_ln2"}));
    
    // Checkpoint each transformer block
    model->addCheckpointBlock({
        prefix + "_attn", prefix + "_ln1",
        prefix + "_ffn1", prefix + "_act",
        prefix + "_ffn2", prefix + "_ln2"
    });
}

model->compile();
model->initialize();
```

## Troubleshooting

### "Cannot add checkpoint block after initialization"
- Ensure `addCheckpointBlock()` is called before `model->initialize()`

### "Layers in checkpoint block must be consecutive"
- Verify that all layers in the block are connected sequentially in the graph

### Memory not reduced as expected
- Check that checkpoint blocks contain at least 2 layers
- Single-layer blocks have overhead that exceeds savings

## Known Issues / TODO

- [ ] **unittest_nntrainer_exe_order**: Requires update for 5-tuple ExecutionOrder format. The test compares execution order output with golden data, which needs regeneration for the new format. (follow-up PR)
- [ ] **Gradient checkpointing unit tests**: Add dedicated unit tests for:
  - CheckpointBlock class functionality
  - Recomputation correctness verification

## See Also

- [NNTrainer Model API](./model-api.md)
- [Memory Management](./memory-management.md)
