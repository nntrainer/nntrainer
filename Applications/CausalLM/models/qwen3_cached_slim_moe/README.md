# Qwen3 Cached Slim MoE Model

This directory contains the implementation for Qwen3 Slim MoE model with caching support.

> 📌 **Note** on `Cached-Slim`: This model extends the Slim approach (dynamic loading) by caching active experts. This strategy minimizes storage I/O bottlenecks, offering a sweet spot between low memory footprint and high inference speed.

## Files
- `qwen3_cached_slim_moe_causallm.cpp`: Cached Slim MoE implementation.
- `qwen_moe_layer_cached.cpp`: Cached MoE layer implementation.
