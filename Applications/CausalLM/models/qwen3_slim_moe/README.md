# Qwen3 Slim MoE Model

This directory contains the implementation for Qwen3 Slim MoE model.
## About the Slim Model
The **Slim** model is designed to minimize peak memory usage by loading experts in an on-the-fly manner.

- **Efficient Initialization**: Instead of loading all model weights at once, the model initializes without heavy expert layers.
- **Dynamic Loading**: Only the activated experts are loaded into memory during runtime.
- **Performance Note**: Since the model dynamically maps memory to experts on storage, inference speed relies heavily on storage read I/O performance.

## Files
- `qwen3_slim_moe_causallm.cpp`: Slim MoE implementation.
- `qwen_moe_layer_fsu.cpp`: FSU optimized layer for Slim MoE.
