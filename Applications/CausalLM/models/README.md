# CausalLM Models

This directory contains the implementations of the Causal Language Models structured into subdirectories.

## Base Model
- **`causal_lm.cpp/h`**: The base class for all CausalLM implementations, defining the core architecture (Embedding, Decoder Blocks, RMSNorm, LMHead).

## Available Models


Here is the list of supported models. We provide **Standard** implementations and **NNTrainer Variants** optimized for on-device environments! 🚀

| Model Name 🏷️ | Size 📏 | Type 🏗️ | Special Features ✨ | Description 📝 |
| :--- | :---: | :---: | :--- | :--- |
| `causal_lm` | - | 📦 Standard | - | Basic implementation of the llama model. |
| `qwen3_causallm` | **0.6B, 1.7B, 4B, 8B, 14B, 32B** | 📦 Standard | - | Basic implementation of the Qwen3 model. |
| `qwen3_moe_causallm` | **30B-A3B** | 📦 Standard | - | Basic implementation of the Qwen3 MoE model. |
| `qwen3_slim_moe_causallm` | **30B-A3B** | 🛠️ **Variant** | 🍃 **Slim** | Activated by FSU scheme (On-the-fly expert loading). |
| `qwen3_cached_slim_moe_causallm` | **30B-A3B** | 🛠️ **Variant** | ⚡ **Cached Slim** | MoE-specific FSU implementation with **expert caching**. |
| `gptoss_causallm` | **20B-A3.6B, 120B-5.1B** | 📦 Standard | - | Basic implementation of the GPT-OSS model. |
| `gptoss_cached_slim_causallm` | **20B-A3.6B, 120B-5.1B** | 🛠️ **Variant** | ⚡ **Cached Slim** | GPT-OSS MoE implementation with **expert caching**. |

> *Note: 📦 **Standard** refers to the basic implementation, while 🛠️ **Variant** refers to models optimized for your device using FSU schemes.*

### MoE inference support

#### 🍃 What is a  `slim` model?
The *_slim_* model reduces peak memory usage by loading experts in an on-the-fly manner.

- **Efficient Initialization**: Instead of loading all model weights at once, the slim model initializes without the heavy expert layers.
- **Dynamic Loading**: Only the activated experts are loaded into memory during runtime, keeping memory usage significantly lower than the original model.
- **Performance Note**: Since the model dynamically maps memory to experts on storage, inference speed relies heavily on the storage read I/O speed.

#### ⚡ What is a `cached` model?

The cached model is a variant of the slim model that caches activated experts. Instead of immediately deactivating experts after use, it delays memory unmapping. This approach reduces repetitive loading overhead, thereby increasing inference speed.

## Directory Structure
Each model directory typically contains:
- `*_causallm.cpp`: The model implementation class.
- `*_layer.cpp`: (Optional) Model-specific custom layer implementations.
- `meson.build`: Build configuration for the model.
- `README.md`: Specific details about the model.
