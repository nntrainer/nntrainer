# GPT-OSS Cached Slim Model

This directory contains the implementation for GPT-OSS Cached Slim model.

> 📌 **Note** on `Cached-Slim`: This model extends the Slim approach (dynamic loading) by caching active experts. This strategy minimizes storage I/O bottlenecks, offering a sweet spot between low memory footprint and high inference speed.

## Files
- `gptoss_cached_slim_causallm.cpp`: Cached Slim GPT-OSS implementation.
- `gpt_oss_moe_layer_cached.cpp`: Cached GPT-OSS MoE layer implementation.
