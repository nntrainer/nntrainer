# Generated Code Pattern Guide

## Overview
The cpp_generator produces model architecture files that follow the EXACT same structure as existing hand-written model files (e.g., `qwen3_causallm.h/cpp`), but GENERATED from the graph IR rather than hand-coded.

## Pattern Structure

### Generated Header File (`{Model}_causallm.h`)
Matches the structure of existing files like `qwen3_causallm.h`:

```cpp
// License header
#ifndef __QWEN3_CAUSAL_LM_H__
#define __QWEN3_CAUSAL_LM_H__ __QWEN3_CAUSAL_LM_H__

#include <causal_lm.h>  // Framework file (NOT model-specific)

namespace causallm {

// Transformer class - handles attention/MLP layers
class Qwen3Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3Transformer";
  
  Qwen3Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}
  
  virtual ~Qwen3Transformer() = default;
  
  Tensor createAttention(...) override;  // Generated from graph IR
  void registerCustomLayers() override;  // Generated from graph IR
};

// CausalLM class - the full model
class Qwen3CausalLM : public CausalLM, public Qwen3Transformer {
public:
  static constexpr const char *architectures = "Qwen3ForCausalLM";
  
  Qwen3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen3Transformer(cfg, generation_cfg, nntr_cfg) {}
  
  virtual ~Qwen3CausalLM() = default;
  
  void registerCustomLayers() override;  // Generated from graph IR
};

} // namespace causallm
#endif
```

### Generated Source File (`{Model}_causallm.cpp`)
Implements the methods using nntrainer layer API:

```cpp
// License header
#include <llm_util.hpp>
#include <model.h>
#include "Qwen3_causallm.h"  // Generated header

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>  // Only if needed

namespace causallm {

// Implements architecture-specific layers
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {
  // Q, K, V projections
  // Normalization (if needed)
  // Attention core with KV cache
  // Output projection
  return attention_output;
}

void Qwen3Transformer::registerCustomLayers() {
  // Register any custom layers (e.g., reshaped_rms_norm)
}

void Qwen3CausalLM::registerCustomLayers() {
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
```

## Key Design Principles

### 1. **NOT a Copy-Paste of Existing Files**
- The cpp_generator extracts architecture STRUCTURE from the graph IR
- It generates the class hierarchy and method signatures
- The layer implementations are built from graph nodes, not hardcoded

### 2. **Portable Across Models**
- `_extract_model_slug(architecture)` extracts the base model name
  - `Qwen3ForCausalLM` → `Qwen3`
  - `Gemma3CausalLM` → `Gemma3`
  - `LlamaForCausalLM` → `Llama`
- Header guards and class names are auto-generated per model
- Works for any HuggingFace model architecture

### 3. **Graph-Based Generation**
- `createAttention()`, `createMLP()`, `createDecoderLayer()` are generated from graph nodes
- Partitions are extracted from graph metadata
- Layer configurations come from model config, not hardcoded

### 4. **Framework Code Unchanged**
- Only generates MODEL-SPECIFIC code:
  - Qwen3Transformer, Qwen3CausalLM classes
  - Per-model attention/MLP implementations
- ALL framework code comes from nntrainer:
  - `causal_lm.h` (base CausalLM class)
  - `transformer.h` (base Transformer class)
  - `llm_util.hpp` (utility functions)
  - Layer implementations (mha_core, reshaped_rms_norm, etc.)

## Installation Directory Structure

```
/Applications/CausalLM/models/qwen3/
├── Qwen3_causallm.h         ← Generated (replaces hand-written)
├── Qwen3_causallm.cpp       ← Generated (replaces hand-written)
├── qwen3_embedding.h        ← Kept as-is (if exists)
├── qwen3_embedding.cpp      ← Kept as-is (if exists)
└── meson.build              ← Updated to include generated files

/Applications/CausalLM/models/
├── causal_lm.h              ← Framework (unchanged)
├── causal_lm.cpp            ← Framework (unchanged)
├── transformer.h            ← Framework (unchanged)
└── transformer.cpp          ← Framework (unchanged)
```

## How cpp_generator Works

1. **Parse Graph IR** → Extract architecture, layers, connections
2. **Determine Model Type** → Check if causallm_component mode
3. **Generate Header**:
   - Create guard macros
   - Define Transformer and CausalLM classes
   - Extract method signatures from graph
4. **Generate Source**:
   - Implement createAttention(), createMLP(), etc. from graph nodes
   - Emit layer creation code using nntrainer API
   - Handle custom layers (e.g., reshaped_rms_norm)
5. **Write Files** with proper filenames: `Qwen3_causallm.h/cpp`

## When Adding New Models

1. User provides a new HuggingFace model
2. Model discovery extracts architecture name (e.g., "Qwen3ForCausalLM")
3. cpp_generator:
   - Extracts model slug: `Qwen3`
   - Generates `Qwen3_causallm.h/cpp`
   - NO manual coding required!
4. Files are installed to `models/qwen3/` (auto-derived)
5. User rebuilds CausalLM project with new model included

## Architecture Flexibility

The generated code handles multiple transformer variants:

| HF Architecture | Generated | Directory | Files |
|---|---|---|---|
| Qwen3ForCausalLM | Qwen3 | models/qwen3/ | Qwen3_causallm.h/cpp |
| Gemma3CausalLM | Gemma3 | models/gemma3/ | Gemma3_causallm.h/cpp |
| LlamaForCausalLM | Llama | models/llama/ | Llama_causallm.h/cpp |
| GPTossForCausalLM | Gptoss | models/gptoss/ | Gptoss_causallm.h/cpp |

Each follows the SAME pattern, just with different model-specific implementations.
