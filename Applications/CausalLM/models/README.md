# CausalLM Models

This directory contains the implementations of the Causal Language Models structured into subdirectories.

## Base Model
- **`causal_lm.cpp/h`**: The base class for all CausalLM implementations, defining the core architecture (Embedding, Decoder Blocks, RMSNorm, LMHead).

## Available Models

| Model Name | Directory | Description |
| :--- | :--- | :--- |
| **Qwen3** | `qwen3` | Standard implementation of Qwen3 Causal LM. |
| **Qwen3 MoE** | `qwen3_moe` | Implementation of Qwen3 Mixture of Experts (MoE) model. |
| **Qwen3 Slim MoE** | `qwen3_slim_moe` | Optimized "Slim" version of Qwen3 MoE using FSU layers. |
| **Qwen3 Cached Slim MoE** | `qwen3_cached_slim_moe` | Cached version of the Slim Qwen3 MoE model for optimized inference. |
| **GPT-OSS** | `gpt_oss` | Implementation of the GPT-OSS model. |
| **GPT-OSS Cached Slim** | `gpt_oss_cached_slim` | Cached Slim version of the GPT-OSS model. |

## Directory Structure
Each model directory typically contains:
- `*_causallm.cpp`: The model implementation class.
- `*_layer.cpp`: (Optional) Model-specific custom layer implementations.
- `meson.build`: Build configuration for the model.
- `README.md`: Specific details about the model.
