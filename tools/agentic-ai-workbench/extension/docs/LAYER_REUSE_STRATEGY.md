# Layer Reuse Strategy

## Overview
Generated model code **reuses pre-built layer implementations** from `/Applications/CausalLM/layers/` instead of duplicating them. This maintains consistency and decorum across all models.

## Layer Library Structure

**Location**: `/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/layers/`

### Available Layers (Pre-built, Tested)

#### Custom Layers (CausalLM-specific)
- `reshaped_rms_norm.h/cpp` - RMS norm with reshaping (for Qwen, Llama, etc.)
- `rms_norm.h/cpp` - Standard RMS normalization
- `rms_reverse_norm.h/cpp` - Reverse RMS norm
- `mha_core.h/cpp` - Multi-Head Attention core (Q/K/V handling)
- `causal_conv1d_layer.h/cpp` - Causal 1D convolution
- `embedding_layer.h/cpp` - Token embedding with optimizations
- `lm_head.h/cpp` - Language model head (logits)
- `logit_softcapping.h/cpp` - Softcapping for logits
- `qkv_layer.h/cpp` - Q/K/V projection layer
- `swiglu.h/cpp` - SwiGLU activation
- `shared_fully_connected_layer.h/cpp` - Shared FC layer
- `custom_multiply.h/cpp` - Custom multiplication
- `scalar_multiply.h/cpp` - Scalar multiplication

#### nntrainer Built-in Layers
- `fully_connected` - Dense/linear layer
- `addition` - Element-wise addition
- `multiply` - Element-wise multiplication
- `layer_norm` - Layer normalization
- `embedding` - Standard embedding

## How Generated Code Uses These Layers

### Pattern 1: Using Built-in Layers
```cpp
// In generated Llama_causallm.cpp

Tensor LlamaTransformer::createAttention(...) {
  // Uses pre-built reshaped_rms_norm layer
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",  // ← Implemented in /layers/reshaped_rms_norm.cpp
    {withKey("name", ...), withKey("feature_size", ...)}
  ));
  Tensor q_normed = q_norm(q);

  // Uses pre-built mha_core layer
  LayerHandle mha(createLayer(
    "mha_core",  // ← Implemented in /layers/mha_core.cpp
    {withKey("num_heads", ...), withKey("num_heads_kv", ...)}
  ));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // Uses built-in fully_connected
  LayerHandle wo(createLayer(
    "fully_connected",  // ← nntrainer built-in
    {withKey("unit", DIM), withKey("disable_bias", "true")}
  ));
  return wo(a);
}
```

### Pattern 2: Registering Custom Layers
```cpp
// In generated Llama_causallm.cpp

void LlamaTransformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));

  try {
    // Register ReshapedRMSNormLayer from /layers/reshaped_rms_norm.cpp
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>
    );
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory: " << e.what() << std::endl;
  }
}
```

## Code Organization

```
Generated Model Code (Llama_causallm.h/cpp)
    ├─ Only contains:
    │  ├─ Class definitions (Llama Transformer, Llama CausalLM)
    │  ├─ Method overrides (createAttention, createMLP, createDecoderLayer)
    │  └─ Layer registration (registerCustomLayers)
    │
    └─ References (does NOT duplicate):
       ├─ Layer implementations from /layers/
       │  ├─ reshaped_rms_norm.cpp
       │  ├─ mha_core.cpp
       │  ├─ embedding_layer.cpp (if needed)
       │  └─ others via createLayer() calls
       │
       └─ Built-in nntrainer layers
          ├─ fully_connected
          ├─ addition
          └─ multiply
```

## Code Reuse Example

### Reference (Qwen3) Implementation
```cpp
// /Applications/CausalLM/models/qwen3/qwen3_causallm.cpp

Tensor Qwen3Transformer::createAttention(...) {
  LayerHandle q_norm(createLayer("reshaped_rms_norm", {...}));
  LayerHandle mha(createLayer("mha_core", {...}));
  LayerHandle wo(createLayer("fully_connected", {...}));
  return wo(a);
}

void Qwen3Transformer::registerCustomLayers() {
  app_context->registerFactory(
    nntrainer::createLayer<causallm::ReshapedRMSNormLayer>
  );
}
```

### Adapted (Llama) Implementation
```cpp
// /Applications/CausalLM/models/llama/Llama_causallm.cpp
// (Adapted from Qwen3 by changing class names only)

Tensor LlamaTransformer::createAttention(...) {
  LayerHandle q_norm(createLayer("reshaped_rms_norm", {...}));  // ← SAME
  LayerHandle mha(createLayer("mha_core", {...}));              // ← SAME
  LayerHandle wo(createLayer("fully_connected", {...}));        // ← SAME
  return wo(a);
}

void LlamaTransformer::registerCustomLayers() {
  app_context->registerFactory(
    nntrainer::createLayer<causallm::ReshapedRMSNormLayer>  // ← SAME
  );
}
```

**No layer implementations duplicated!** ✅

## Benefits

| Aspect | Without Reuse | With Reuse |
|---|---|---|
| **Code Duplication** | Layer code in every model | Single copy in /layers/ |
| **Consistency** | Each model might implement differently | All models use same layers |
| **Maintenance** | Fix bug in N places | Fix once in /layers/ |
| **Testing** | Test each model's layers | Test /layers/ once |
| **Updates** | Update N model files | Update /layers/ only |
| **Size** | Large (repeated code) | Small (only references) |

## Model-Specific vs Layer-Specific

### What IS Model-Specific (Generated)
```cpp
class LlamaTransformer : virtual public Transformer {
  Tensor createAttention(const int layer_id, int seq_len, ...) override;
  Tensor createMLP(const int layer_id, int seq_len, ...) override;
  void registerCustomLayers() override;
};
```

### What IS NOT Model-Specific (Reused)
```cpp
// In createAttention():
LayerHandle q_norm(createLayer("reshaped_rms_norm", {...}));
// ↓ Uses ReshapedRMSNormLayer from /layers/reshaped_rms_norm.cpp

LayerHandle mha(createLayer("mha_core", {...}));
// ↓ Uses MHACoreLayer from /layers/mha_core.cpp

LayerHandle wo(createLayer("fully_connected", {...}));
// ↓ Uses nntrainer's built-in FullyConnectedLayer
```

## Decorum (Consistency Standard)

**All models follow the same pattern**:
1. Model-specific: Class definitions and method signatures
2. Layer usage: Call createLayer() with parameters
3. Layer registration: Register custom layers via registerFactory()
4. Layer implementations: ALL come from /layers/ or nntrainer built-ins

**Result**: Clean separation, easy to maintain, consistent across all models.

## Header Comment in Generated Files

Every adapted model file gets this notice:

```cpp
/*
 * LAYER REUSE NOTICE:
 * This Llama implementation reuses tested layer implementations from:
 * /Applications/CausalLM/layers/
 *
 * Layer implementations used (NOT duplicated):
 * - reshaped_rms_norm (reshaped_rms_norm.cpp)
 * - mha_core (mha_core.cpp)
 * - fully_connected (nntrainer built-in)
 * - addition (nntrainer built-in)
 *
 * ONLY model architecture code (class methods) is in this file.
 * All layer implementations are referenced from CausalLM/layers/ folder.
 */
```

## Adding New Layers

If a model needs a new layer type:

1. **Don't duplicate** in the generated model file
2. **Create** `/Applications/CausalLM/layers/new_layer.h/cpp`
3. **Register** in model's `registerCustomLayers()`
4. **Use** via `createLayer("new_layer", {...})` in all models

Example:
```cpp
// layers/my_custom_layer.h/cpp (once, tested)
class MyCustomLayer : public nntrainer::Layer { ... };

// models/qwen3/qwen3_causallm.cpp (and all other models)
void Qwen3Transformer::registerCustomLayers() {
  app_context->registerFactory(
    nntrainer::createLayer<causallm::MyCustomLayer>
  );
}

// In createAttention/createMLP:
LayerHandle custom(createLayer("my_custom_layer", {...}));
```

All models automatically get access to the new layer! ✅

---

**Status**: ✅ Layer reuse strategy fully implemented
- Generated code references /layers/ implementations
- No layer duplication
- Consistent across all models
- Easy to maintain and update
