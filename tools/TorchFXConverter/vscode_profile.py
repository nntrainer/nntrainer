#!/usr/bin/env python3
"""VS Code bridge for profiling PyTorch models.

Supports:
  - HuggingFace models (--model)
  - Local .py files with nn.Module classes (--local-model --class-name)

Uses torch.profiler to capture per-module CPU/CUDA timing and outputs
a profile_result.json compatible with the Graph Visualizer.
"""

import sys
import os
import json
import argparse
import importlib.util
import inspect
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Local .py model loading (shared logic with vscode_bridge.py)
# ---------------------------------------------------------------------------

def load_local_model(py_path, class_name):
    """Dynamically load an nn.Module class from a local .py file."""
    module_dir = os.path.dirname(os.path.abspath(py_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        candidates = [
            name for name, obj in inspect.getmembers(mod)
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module
        ]
        raise ValueError(
            f"Class '{class_name}' not found in {py_path}. "
            f"Available nn.Module classes: {candidates}"
        )

    cls = getattr(mod, class_name)
    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        raise ValueError(f"'{class_name}' is not an nn.Module subclass")

    return cls, mod


def infer_constructor_args(cls):
    """Try to infer reasonable default constructor arguments."""
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            guesses = {
                "vocab_size": 1000, "num_embeddings": 1000,
                "hidden_size": 256, "d_model": 256, "embed_dim": 256,
                "num_heads": 4, "nhead": 4, "num_attention_heads": 4,
                "num_layers": 2, "num_hidden_layers": 2,
                "intermediate_size": 512, "dim_feedforward": 512,
                "num_classes": 10, "output_size": 10,
                "input_size": 256, "in_features": 256, "in_channels": 3,
                "out_features": 256, "out_channels": 64,
                "kernel_size": 3, "dropout": 0.0,
                "max_seq_len": 512, "max_position_embeddings": 512,
                "bias": True, "batch_first": True,
            }
            if name in guesses:
                kwargs[name] = guesses[name]
            else:
                raise ValueError(
                    f"Cannot infer constructor arg '{name}' for {cls.__name__}. "
                    f"Provide --input-desc with constructor args."
                )
    return kwargs


def build_trace_inputs(model, input_desc, seq_len):
    """Build trace input tensors from user description or by inference."""
    if input_desc:
        desc = json.loads(input_desc)

        if "trace_inputs" in desc:
            inputs = {}
            for key, shape in desc["trace_inputs"].items():
                if "int" in key or "ids" in key or "mask" in key:
                    inputs[key] = torch.randint(0, 100, shape)
                else:
                    inputs[key] = torch.randn(*shape)
            return inputs

        # Assume desc is {name: shape} for trace inputs
        inputs = {}
        for key, shape in desc.items():
            if "int" in key or "ids" in key or "mask" in key:
                inputs[key] = torch.randint(0, 100, shape)
            else:
                inputs[key] = torch.randn(*shape)
        return inputs

    # Auto-detect from forward() signature
    sig = inspect.signature(model.forward)
    inputs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in ("input_ids", "src", "tgt"):
            inputs[name] = torch.randint(0, 100, (1, seq_len))
        elif name in ("attention_mask", "src_mask", "mask"):
            inputs[name] = torch.ones(1, seq_len, dtype=torch.long)
        elif name in ("x", "input", "inputs"):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                in_dim = first_param.shape[-1] if first_param.dim() >= 2 else first_param.shape[0]
                inputs[name] = torch.randn(1, seq_len, in_dim)
            else:
                inputs[name] = torch.randn(1, seq_len, 256)
        elif param.default is inspect.Parameter.empty:
            inputs[name] = torch.randn(1, seq_len, 256)

    if not inputs:
        inputs = {"x": torch.randn(1, seq_len, 256)}

    return inputs


# ---------------------------------------------------------------------------
# Profiling logic
# ---------------------------------------------------------------------------

def profile_model(model, inputs, num_runs=5, num_warmup=2):
    """Profile a model using torch.profiler and return per-module timing data.

    Returns a dict matching the ProfileData format expected by the visualizer.
    """
    model.eval()

    # Move model and inputs to same device
    device = next(model.parameters(), torch.tensor(0)).device
    moved_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            moved_inputs[k] = v.to(device)
        else:
            moved_inputs[k] = v
    inputs = moved_inputs

    # Warmup runs
    print(f"PROGRESS: Warming up ({num_warmup} runs)...", flush=True)
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)

    # Profile with torch.profiler
    print(f"PROGRESS: Profiling ({num_runs} runs)...", flush=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_start = time.perf_counter()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)

    total_elapsed = (time.perf_counter() - total_start) * 1000  # ms

    # Parse profiler events to extract per-module timing
    module_times = {}  # module_name -> {cpu_ms, cuda_ms, count, calls}

    for event in prof.key_averages():
        key = event.key
        cpu_us = event.cpu_time_total
        cuda_us = event.cuda_time_total if hasattr(event, 'cuda_time_total') else 0
        count = event.count

        # Try to get the module path from the event
        # torch.profiler events with with_modules=True have module hierarchy
        module_path = ""
        if hasattr(event, 'module_stack') and event.module_stack:
            # module_stack is a tuple of (scope_name, ...)
            parts = [s for s in event.module_stack if s]
            if parts:
                module_path = ".".join(parts)

        if not module_path:
            continue

        if module_path not in module_times:
            module_times[module_path] = {
                'cpu_ms': 0,
                'cuda_ms': 0,
                'count': 0,
                'calls': 0,
            }

        module_times[module_path]['cpu_ms'] += cpu_us / 1000.0
        module_times[module_path]['cuda_ms'] += cuda_us / 1000.0
        module_times[module_path]['count'] += 1
        module_times[module_path]['calls'] += count

    # Also extract from events table (more reliable for leaf modules)
    # Use prof.table() to get a sorted table and parse key_averages
    event_list = prof.key_averages()
    module_data = {}

    for evt in event_list:
        # Get module stack path
        mod_stack = getattr(evt, 'module_stack', None) or getattr(evt, 'module', None)
        if not mod_stack:
            continue

        # Convert module stack to string path
        if isinstance(mod_stack, (list, tuple)):
            mod_path = ".".join(str(s) for s in mod_stack if s)
        else:
            mod_path = str(mod_stack)

        if not mod_path:
            continue

        cpu_ms = evt.cpu_time_total / 1000.0
        cuda_ms = getattr(evt, 'cuda_time_total', 0) / 1000.0
        self_cpu_ms = evt.self_cpu_time_total / 1000.0
        count = evt.count

        # Use the most specific time (self time to avoid double-counting)
        if mod_path not in module_data:
            module_data[mod_path] = {
                'cpu_ms': 0,
                'cuda_ms': 0,
                'self_cpu_ms': 0,
                'count': 0,
                'calls': 0,
            }

        module_data[mod_path]['cpu_ms'] += cpu_ms
        module_data[mod_path]['cuda_ms'] += cuda_ms
        module_data[mod_path]['self_cpu_ms'] += self_cpu_ms
        module_data[mod_path]['count'] += 1
        module_data[mod_path]['calls'] += count

    # Build layers list from module_data
    # Use self_cpu_ms for accurate per-layer timing (avoids double-counting)
    total_self_cpu = sum(d['self_cpu_ms'] for d in module_data.values()) or 1.0

    layers = []
    for mod_path, data in sorted(module_data.items(), key=lambda x: -x[1]['self_cpu_ms']):
        if data['self_cpu_ms'] < 0.001:
            continue  # Skip negligible entries

        pct = (data['self_cpu_ms'] / total_self_cpu) * 100.0
        time_ms = data['self_cpu_ms'] / num_runs  # Average per run

        # Extract module type from the path
        parts = mod_path.split('.')
        layer_type = parts[-1] if parts else mod_path

        layers.append({
            'name': mod_path,
            'layer_type': layer_type,
            'time_ms': round(time_ms, 4),
            'memory_mb': 0,
            'flops': 0,
            'pct_of_total': round(pct, 2),
            'avg_ms': round(time_ms, 4),
            'min_ms': 0,
            'max_ms': 0,
            'count': data['calls'] // max(num_runs, 1),
        })

    # Identify bottlenecks (>10% of total)
    bottlenecks = [l['name'] for l in layers if l['pct_of_total'] > 10]

    # Get memory info
    memory_mb = 0
    try:
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    result = {
        'model_name': getattr(model, '__class__', type(model)).__name__,
        'total_time_ms': round(total_elapsed / num_runs, 2),
        'total_memory_mb': round(memory_mb, 2),
        'num_runs': num_runs,
        'seq_len': 0,
        'layers': layers,
        'bottlenecks': bottlenecks,
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile a PyTorch model for the Graph Visualizer")

    # HuggingFace model mode
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID or local model path")

    # Local .py file mode
    parser.add_argument("--local-model", default=None,
                        help="Path to local .py file containing nn.Module")
    parser.add_argument("--class-name", default=None,
                        help="nn.Module class name in the .py file")
    parser.add_argument("--input-desc", default=None,
                        help="JSON describing input shapes or constructor args")

    # Common options
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of profiling runs")
    parser.add_argument("--num-warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    if not args.model and not args.local_model:
        parser.error("Either --model or --local-model is required")

    os.makedirs(args.output, exist_ok=True)

    if args.local_model:
        # ---- Local .py model mode ----
        print(f"PROGRESS: Loading local model from {args.local_model}...", flush=True)

        cls, mod = load_local_model(args.local_model, args.class_name)

        # Parse input-desc for possible constructor args
        constructor_kwargs = {}
        input_desc_for_trace = args.input_desc
        if args.input_desc:
            try:
                desc = json.loads(args.input_desc)
                if "constructor_args" in desc:
                    constructor_kwargs = desc["constructor_args"]
                    input_desc_for_trace = json.dumps(desc.get("trace_inputs", {})) if "trace_inputs" in desc else None
            except json.JSONDecodeError:
                pass

        if constructor_kwargs:
            model = cls(**constructor_kwargs)
        else:
            try:
                model = cls()
            except TypeError:
                print("PROGRESS: Inferring constructor arguments...", flush=True)
                constructor_kwargs = infer_constructor_args(cls)
                model = cls(**constructor_kwargs)

        model.eval()
        print(f"PROGRESS: Model loaded: {args.class_name} "
              f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)", flush=True)

        inputs = build_trace_inputs(model, input_desc_for_trace, args.seq_len)

    elif args.model:
        # ---- HuggingFace model mode ----
        print(f"PROGRESS: Loading HuggingFace model: {args.model}...", flush=True)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("ERROR: transformers package required for HuggingFace models", file=sys.stderr)
            sys.exit(1)

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        except Exception as e:
            # Try as local path
            if os.path.isdir(args.model):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.model)
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                    )
                except Exception as e2:
                    print(f"ERROR: Failed to load model: {e2}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
                sys.exit(1)

        model.eval()
        print(f"PROGRESS: Model loaded "
              f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)", flush=True)

        # Build inputs from tokenizer
        seq_len = args.seq_len
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # ---- Profile the model ----
    print(f"PROGRESS: Starting profiling ({args.num_runs} runs)...", flush=True)
    result = profile_model(model, inputs, num_runs=args.num_runs, num_warmup=args.num_warmup)

    # Write result
    result_path = os.path.join(args.output, "profile_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"PROGRESS: Profile complete. {len(result['layers'])} modules profiled.", flush=True)
    print(f"PROGRESS: Total time: {result['total_time_ms']:.2f} ms (avg of {result['num_runs']} runs)", flush=True)
    if result['bottlenecks']:
        print(f"PROGRESS: Bottlenecks: {', '.join(result['bottlenecks'][:5])}", flush=True)


if __name__ == "__main__":
    main()
