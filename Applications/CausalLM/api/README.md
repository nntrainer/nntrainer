# CausalLM API

This directory contains the C API for CausalLM application, designed to provide a simple interface for loading and running Large Language Models (LLMs) on various backends, including Android.

## Overview

The API provides functionality to:
- Initialize and configure the CausalLM environment.
- Load pre-trained models with specific quantization settings.
- Run inference (text generation) given a prompt.
- Retrieve performance metrics (token counts, duration).

## Build & Integration

The CausalLM API is built as a separate shared library `libcausallm_api.so`, which depends on the core logic in `libcausallm_core.so`.

### Build Artifacts (Android)

- **`libcausallm_core.so`**: Contains the core LLM implementation (Model, Layers, etc.).
- **`libcausallm_api.so`**: Contains the C API implementation (`causal_lm_api.cpp`) and configuration helpers (`model_config.cpp`).

### Linking

When integrating this API into your application (e.g., via JNI), you must link against both libraries:
1.  `libcausallm_api.so`
2.  `libcausallm_core.so`
3.  `libnntrainer.so` (Dependency)

## Directory Structure & Model Loading

The API strictly relies on registered model types and quantization settings to locate model files. There are two modes of loading, depending on how the model is registered within the library.

**Path Convention:** `./models/{ModelKey}{QuantizationSuffix}/`

- **ModelKey**: Derived from `ModelType` (e.g., `qwen3-0.6b`).
- **QuantizationSuffix**: Derived from `ModelQuantizationType` (e.g., `-w16a16`).

### 1. Internal/Embedded Configuration (Pre-configured)

Some model configurations (including architecture, tokenizer settings, and generation parameters) are embedded directly into the CausalLM library code (via `model_config.cpp`). This protects the model specifications and simplifies deployment.

- **Required Files:**
    - **Weight Binary File**: The actual model weights (e.g., `qwen3-0.6b-fp32.bin`). The filename is hardcoded in the internal configuration.
    - **Tokenizer Files**: `tokenizer.json` / `vocab.json` (if required by the specific tokenizer implementation).

- **Ignored Files:**
    - `config.json`, `nntr_config.json`, `generation_config.json` are **NOT** loaded from the disk even if they exist.

### 2. External/File-based Configuration

For registered model types that do not have a specific hardcoded configuration for the requested quantization, the API falls back to loading configuration files from the directory.

- **Required Files:**
    - **`config.json`**: Model architecture configuration (HuggingFace format).
    - **`nntr_config.json`**: NNTrainer specific configuration.
        - Must contain `"model_file_name"` field pointing to the binary weight file.
    - **Weight Binary File**: The file specified in `nntr_config.json`.
    - **`generation_config.json`**: (Optional) Generation parameters.
    - **Tokenizer Files**: `tokenizer.json` / `vocab.json`.

**Note:** When `debug_mode` is enabled in `setOptions`, the API will attempt to validate the existence of the required files for the resolved mode during initialization.

## API Reference

The main header file is `causal_lm_api.h`.

### Enums

#### `ErrorCode`
Return codes for API functions.
- `CAUSAL_LM_ERROR_NONE`: Success.
- `CAUSAL_LM_ERROR_INVALID_PARAMETER`: Invalid argument provided.
- `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED`: Failed to load the model.
- `CAUSAL_LM_ERROR_INFERENCE_FAILED`: Inference execution failed.
- `CAUSAL_LM_ERROR_NOT_INITIALIZED`: API not initialized or model not loaded.
- `CAUSAL_LM_ERROR_INFERENCE_NOT_RUN`: Metrics requested before inference run.

#### `BackendType`
Target backend for computation.
- `CAUSAL_LM_BACKEND_CPU`: CPU execution (default).
- `CAUSAL_LM_BACKEND_GPU`: GPU execution (Planned).
- `CAUSAL_LM_BACKEND_NPU`: NPU execution (Planned).

#### `ModelType`
Supported pre-defined model types.
- `CAUSAL_LM_MODEL_QWEN3_0_6B`: Qwen3 0.6B model.

#### `ModelQuantizationType`
Supported quantization formats.
- `CAUSAL_LM_QUANTIZATION_W4A32`: 4-bit weights, 32-bit activations.
- `CAUSAL_LM_QUANTIZATION_W16A16`: 16-bit weights, 16-bit activations (FP16).
- `CAUSAL_LM_QUANTIZATION_W8A16`: 8-bit weights, 16-bit activations.
- `CAUSAL_LM_QUANTIZATION_W32A32`: 32-bit weights, 32-bit activations (FP32).

### Functions

#### `ErrorCode setOptions(Config config)`
Sets global configuration options.
- **config.use_chat_template**: Whether to apply chat template in `runModel()`.
- **config.debug_mode**: Validate model files during initialization.
- **config.verbose**: Print output during generation.
- **config.chat_template_name**: Template name to select when `chat_template` is an array (e.g., `"default"`, `"tool_use"`). `NULL` defaults to `"default"`.

#### `ErrorCode loadModel(BackendType compute, ModelType modeltype, ModelQuantizationType quant_type)`
Loads a registered model.
- **compute**: Backend to use.
- **modeltype**: Specific model enum.
- **quant_type**: Quantization type.

#### `ErrorCode runModel(const char *inputTextPrompt, const char **outputText)`
Runs inference on the loaded model.
- **inputTextPrompt**: The input text/prompt.
- **outputText**: Pointer to store the result string.

#### `ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics)`
Retrieves performance metrics of the last run.
- **metrics**: Pointer to `PerformanceMetrics` struct to be filled.
- `prefill_tokens`, `prefill_duration_ms`: Stats for prompt processing.
- `generation_tokens`, `generation_duration_ms`: Stats for token generation.
- `total_duration_ms`: Total execution time from start to finish.
- `peak_memory_kb`: Peak resident set size (memory usage) in KB.

#### `ErrorCode applyChatTemplate(const CausalLMChatMessage *messages, size_t num_messages, bool add_generation_prompt, const char **formattedText)`
Applies chat template to messages without running inference. Useful for debugging or verifying prompt formatting.
- **messages**: Array of `CausalLMChatMessage` structs (role + content).
- **num_messages**: Number of messages in the array.
- **add_generation_prompt**: Whether to append generation prompt (e.g., `<|im_start|>assistant\n`).
- **formattedText**: Pointer to store the formatted prompt string.

#### `ErrorCode runModelWithMessages(const CausalLMChatMessage *messages, size_t num_messages, bool add_generation_prompt, const char **outputText)`
Applies chat template to messages and runs inference. Internally calls `applyChatTemplate()` then `runModel()`.
- **messages**: Array of `CausalLMChatMessage` structs (role + content).
- **num_messages**: Number of messages in the array.
- **add_generation_prompt**: Whether to append generation prompt.
- **outputText**: Pointer to store the generated output string.

## Usage Examples

### 1. Single Prompt

```c
#include "causal_lm_api.h"
#include <stdio.h>

int main() {
    // 1. Set Options
    Config config;
    config.use_chat_template = true;
    config.debug_mode = false;
    setOptions(config);

    // 2. Load Model
    // Automatically looks for files in "./models/qwen3-0.6b-w16a16/"
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU, 
                              CAUSAL_LM_MODEL_QWEN3_0_6B, 
                              CAUSAL_LM_QUANTIZATION_W16A16);
    
    if (err != CAUSAL_LM_ERROR_NONE) {
        printf("Failed to load model\n");
        return -1;
    }

    // 3. Run Inference
    const char* output = NULL;
    err = runModel("Hello, how are you?", &output);
    
    if (err == CAUSAL_LM_ERROR_NONE) {
        printf("Response: %s\n", output);
    }

    // 4. Check Metrics
    PerformanceMetrics metrics;
    if (getPerformanceMetrics(&metrics) == CAUSAL_LM_ERROR_NONE) {
        printf("Generated %d tokens in %.2f ms\n", 
               metrics.generation_tokens, metrics.generation_duration_ms);
    }

    return 0;
}
```

### 2. Chat Template with Messages

```c
#include "causal_lm_api.h"
#include <stdio.h>

int main() {
    Config config = {.use_chat_template = true, .debug_mode = false, .verbose = true};
    setOptions(config);

    loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
              CAUSAL_LM_QUANTIZATION_W16A16);

    // System prompt + user message
    CausalLMChatMessage msgs[] = {
        {"system", "You are a helpful AI assistant."},
        {"user",   "What is machine learning?"}
    };

    const char *output;
    runModelWithMessages(msgs, 2, true, &output);
    printf("Response: %s\n", output);

    return 0;
}
```

### 3. Conversation History

```c
    // Multiple user/assistant turns
    CausalLMChatMessage msgs[] = {
        {"user",      "Hello, how are you?"},
        {"assistant", "I'm doing great. How can I help you today?"},
        {"user",      "What is deep learning?"},
        {"assistant", "Deep learning uses neural networks with many layers."},
        {"user",      "How is it different from traditional ML?"}
    };

    const char *output;
    runModelWithMessages(msgs, 5, true, &output);
    printf("Response: %s\n", output);
```

### 4. Preview Formatted Prompt (No Inference)
```c
    CausalLMChatMessage msgs[] = {
        {"system", "You are a helpful assistant."},
        {"user",   "Hello!"}
    };
    const char *formatted;
    applyChatTemplate(msgs, 2, true, &formatted);
    printf("Formatted prompt:\n%s\n", formatted);
    // Output (Qwen3):
    // <|im_start|>system
    // You are a helpful assistant.<|im_end|>
    // <|im_start|>user
    // Hello!<|im_end|>
    // <|im_start|>assistant
```

### 5. Named Template Selection (default / tool_use)

When `tokenizer_config.json` contains an array of named templates, select by name:

```c
    // Use "default" template (normal conversation)
    Config config;
    config.use_chat_template = true;
    config.chat_template_name = "default";
    setOptions(config);
    loadModel(...);

    // Use "tool_use" template (function calling)
    Config config;
    config.use_chat_template = true;
    config.chat_template_name = "tool_use";
    setOptions(config);
    loadModel(...);
```

Supported by models like Gemma3 that provide multiple templates:
```json
"chat_template": [
    {"name": "default",  "template": "..."},
    {"name": "tool_use", "template": "..."}
]
```

If the requested name is not found, falls back to the first template with a warning.
