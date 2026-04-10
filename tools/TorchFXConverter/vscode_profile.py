#!/usr/bin/env python3
"""Profile a PyTorch model layer-by-layer using torch.profiler.
Outputs JSON with per-module timing, memory, and bottleneck analysis."""

import sys
import os
import json
import argparse
import time

import torch
import torch.nn as nn


def profile_model(model, input_kwargs, num_runs=5, warmup=2):
    """Profile model with torch.profiler and extract per-module stats."""
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

    # Move inputs to device
    for k in input_kwargs:
        if isinstance(input_kwargs[k], torch.Tensor):
            input_kwargs[k] = input_kwargs[k].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(**input_kwargs)

    # Method 1: Hook-based per-module timing (more reliable for per-layer)
    module_times = {}
    module_memory = {}
    hooks = []

    def make_hooks(name, mod):
        times = []
        mem = []

        def fwd_pre(m, inp):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter())
            if torch.cuda.is_available():
                mem.append(torch.cuda.memory_allocated())
            else:
                mem.append(0)

        def fwd_post(m, inp, out):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = times.pop()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            mem_before = mem.pop()
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                mem_delta = (mem_after - mem_before) / (1024 * 1024)  # MB
            else:
                mem_delta = 0.0

            if name not in module_times:
                module_times[name] = []
                module_memory[name] = []
            module_times[name].append(elapsed)
            module_memory[name].append(mem_delta)

        hooks.append(mod.register_forward_pre_hook(fwd_pre))
        hooks.append(mod.register_forward_hook(fwd_post))

    # Register hooks on leaf modules only
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # leaf module
            make_hooks(name, mod)

    # Run profiling
    total_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(**input_kwargs)
    total_time = (time.perf_counter() - total_start) * 1000 / num_runs  # avg ms

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate results
    layer_profiles = []
    total_layer_time = 0
    for name in module_times:
        times = module_times[name]
        mems = module_memory[name]
        avg_time = sum(times) / len(times)
        avg_mem = sum(mems) / len(mems) if mems else 0
        total_layer_time += avg_time
        mod = dict(model.named_modules()).get(name)
        layer_type = type(mod).__name__ if mod else "unknown"

        # Estimate FLOPs for linear layers
        flops = 0
        if mod and isinstance(mod, nn.Linear):
            flops = 2 * mod.in_features * mod.out_features
        elif mod and isinstance(mod, nn.Embedding):
            flops = mod.embedding_dim

        layer_profiles.append({
            "name": name,
            "layer_type": layer_type,
            "time_ms": round(avg_time, 4),
            "memory_mb": round(avg_mem, 4),
            "flops": flops,
            "pct_of_total": 0,  # filled below
        })

    # Calculate percentages and sort
    for lp in layer_profiles:
        lp["pct_of_total"] = round(lp["time_ms"] / total_layer_time * 100, 2) if total_layer_time > 0 else 0

    layer_profiles.sort(key=lambda x: x["time_ms"], reverse=True)

    # Identify bottlenecks (top 20% by time, or > 10% of total)
    bottlenecks = []
    for lp in layer_profiles:
        if lp["pct_of_total"] >= 10 or layer_profiles.index(lp) < max(1, len(layer_profiles) // 5):
            bottlenecks.append(lp["name"])

    total_memory = sum(lp["memory_mb"] for lp in layer_profiles)

    return {
        "total_time_ms": round(total_time, 2),
        "total_memory_mb": round(total_memory, 4),
        "num_runs": num_runs,
        "layers": layer_profiles,
        "bottlenecks": bottlenecks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("PROGRESS: Loading model for profiling...", flush=True)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoModel

    try:
        config = AutoConfig.from_pretrained(args.model)
    except Exception:
        config_path = os.path.join(args.model, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = type("Cfg", (), json.load(f))()
        else:
            raise

    model_type = getattr(config, "model_type", "")
    causal_types = {
        "qwen3", "qwen2", "llama", "mistral", "gpt2", "gpt_neo", "gpt_neox",
        "phi", "gemma", "gemma2", "gemma3_text", "gemma3n_text", "starcoder2",
        "codegen", "lfm2", "granitemoehybrid", "mamba", "mamba2",
    }
    encoder_decoder_types = {
        "t5", "mt5", "bart", "mbart", "pegasus", "marian", "t5gemma2",
    }
    is_encoder_decoder = model_type in encoder_decoder_types

    try:
        if model_type in causal_types:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        else:
            model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    except (OSError, ValueError):
        if model_type in causal_types:
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModel.from_config(config)

    # Ensure model is float32 for profiling
    model = model.float()
    model.eval()

    vocab_size = getattr(config, "vocab_size", 30000)
    input_kwargs = {"input_ids": torch.randint(0, vocab_size, (1, args.seq_len))}
    if is_encoder_decoder:
        input_kwargs["decoder_input_ids"] = torch.randint(
            0, min(vocab_size, 1000), (1, max(1, args.seq_len // 2)))

    model_name = args.model.replace("/", "_").replace("-", "_")

    print("PROGRESS: Running profiling (" + str(args.num_runs) + " iterations)...", flush=True)

    result = profile_model(model, input_kwargs, num_runs=args.num_runs)
    result["model_name"] = model_name
    result["seq_len"] = args.seq_len

    out_path = os.path.join(args.output, "profile_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("PROGRESS: Done. " + str(len(result['layers'])) + " layers profiled.", flush=True)
    print("PROGRESS: Total time: " + str(result['total_time_ms']) + " ms", flush=True)
    print("PROGRESS: Bottlenecks: " + str(result['bottlenecks'][:5]), flush=True)


if __name__ == "__main__":
    main()
