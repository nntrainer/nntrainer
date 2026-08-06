#!/usr/bin/env python3
"""
nntrainer benchmark for CausalLM models with configuration sweeping.

Usage:
  python3 benchmark_android.py -m <model_path> [options]

This script can sweep through multiple configurations:
  - Different thread counts: -t 1,2,4,8
  - Different generation lengths: -n 128,512,1024
  - Different prompt lengths: -p 256,512,1024

Example:
  python3 benchmark_android.py -m /data/local/tmp/Quick.AI/models/qwen3-0.6b -t 1,2,4,8 -n 128,256
"""

import subprocess
import re
import time
import statistics
import sys
import argparse
import json
import tempfile
import os
import shutil
from itertools import product
from tabulate import tabulate
from transformers import AutoTokenizer

from device_utils import (
    get_thermal_temp, 
    wait_for_cooling,
    get_device_model,
    get_model_size,
) 

def generate_sample_input(target_tokens, local_tokenizer_path=None):
    """
    Generate sample input that matches target token count.
    If transformers is available, use exact tokenizer. Otherwise, use heuristic.
    """
    if local_tokenizer_path:
        # Load tokenizer from local path
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(local_tokenizer_path))

        # Generic base text (repeating pattern)
        base_token = 5555
        base_text = tokenizer.decode([base_token])

        generated_text = base_text * target_tokens
        
        return generated_text
    else:
        # Heuristic fallback: assume ~4 chars per token on average
        chars_per_token = 4
        target_chars = target_tokens * chars_per_token
        
        # Use a repeating pattern
        base_text = "The quick brown fox jumps over the lazy dog. "
        repeats = max(1, target_chars // len(base_text) + 1)
        generated_text = base_text * repeats
        
        # Trim to approximate length
        return generated_text[:target_chars]


def backup_and_modify_config(model_path, n_prompt, n_gen, batch_size=1):
    """
    Backup original nntr_config.json from device and create modified version.
    Returns context manager that restores original config on exit.
    """
    class ConfigModifier:
        def __init__(self, model_path, n_prompt, n_gen, batch_size):
            self.n_prompt = n_prompt
            self.n_gen = n_gen
            self.batch_size = batch_size
            self.device_backup = None
            self.temp_config_path = None
            self.device_config_path = f"{model_path}/nntr_config.json"
            
        def __enter__(self):
            # Backup device config
            result = subprocess.run(
                ["adb", "shell", "cat", self.device_config_path],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Could not read config from device: {result.stderr}")
            
            self.device_backup = result.stdout
            
            # Create backup on device
            subprocess.run(
                ["adb", "shell", "cp", self.device_config_path, self.device_config_path + ".benchmark_backup"],
                capture_output=True
            )
            
            # Load and modify config
            config = json.loads(self.device_backup)
            config["init_seq_len"] = self.n_prompt
            config["num_to_generate"] = self.n_gen
            config["batch_size"] = self.batch_size
            
            # Generate sample_input matching target token count
            local_tokenizer_path = None
            
            if "tokenizer_file" in config:
                device_tokenizer_path = config["tokenizer_file"]
                
                # Create local temp directory for tokenizer
                temp_dir = tempfile.mkdtemp(prefix="tokenizer_")
                
                try:
                    # Extract tokenizer directory name from device path
                    tokenizer_dir = os.path.dirname(device_tokenizer_path)
                    tokenizer_filename = os.path.basename(device_tokenizer_path)
                    
                    # Pull tokenizer directory from device
                    print(f"  Pulling tokenizer from device...")
                    result = subprocess.run(
                        ["adb", "pull", tokenizer_dir + '/' + tokenizer_filename, temp_dir],
                        capture_output=True, text=True
                    )
                    result = subprocess.run(
                        ["adb", "pull", tokenizer_dir + '/' + 'config.json', temp_dir],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        local_tokenizer_path = os.path.join(temp_dir, tokenizer_filename)
                    else:
                        print(f"  Warning: Could not pull tokenizer, using heuristic")
                        shutil.rmtree(temp_dir)
                        temp_dir = None
                except Exception as e:
                    print(f"  Warning: Could not pull tokenizer: {e}")
                    if temp_dir:
                        shutil.rmtree(temp_dir)
                    temp_dir = None
            
            generated_input = generate_sample_input(self.n_prompt, local_tokenizer_path)
            config["sample_input"] = generated_input
            
            if local_tokenizer_path:
                print(f"Generated sample_input ({self.n_prompt} token length, using tokenizer)")
            else:
                print(f"Generated sample_input ({self.n_prompt} token length, heuristic)")
            
            # Clean up temporary tokenizer directory
            if local_tokenizer_path and os.path.exists(os.path.dirname(local_tokenizer_path)):
                shutil.rmtree(os.path.dirname(local_tokenizer_path))
            
            # Create temporary file with modified config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2)
                self.temp_config_path = f.name
            
            # Push modified config to device
            result = subprocess.run(
                ["adb", "push", self.temp_config_path, self.device_config_path],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Could not push config to device: {result.stderr}")
            
            return config
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore device config from backup
            if self.device_backup:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(self.device_backup)
                    temp_backup_path = f.name
                
                try:
                    subprocess.run(
                        ["adb", "push", temp_backup_path, self.device_config_path],
                        capture_output=True
                    )
                    os.remove(temp_backup_path)
                except Exception as e:
                    print(f"Warning: Could not restore config: {e}")
            
            # Clean up temporary files
            if self.temp_config_path and os.path.exists(self.temp_config_path):
                os.remove(self.temp_config_path)
            
            # Remove backup from device
            subprocess.run(
                ["adb", "shell", "rm", "-f", self.device_config_path + ".benchmark_backup"],
                capture_output=True
            )
            
            return False
    
    return ConfigModifier(model_path, n_prompt, n_gen, batch_size)


def run_single_trial(model_path, omp_threads=None):
    """Run a single benchmark trial and collect metrics."""
    # Build command to run nntrainer C++ binary
    # Set OMP_NUM_THREADS as environment variable for the shell command
    if omp_threads:
        cmd = [
            "adb", "shell",
            f"cd /data/local/tmp/Quick.AI && OMP_NUM_THREADS={omp_threads} ./run_causallm.sh '{model_path}'"
        ]
    else:
        cmd = [
            "adb", "shell",
            f"cd /data/local/tmp/Quick.AI && ./run_causallm.sh '{model_path}'"
        ]
    
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    print(output)
    
    # Parse TPS from output
    prefill_match = re.search(r"prefill:.*,\s+([\d\.]+)\s+TPS", output)
    
    # Parse generation TPS if available
    gen_match = re.search(r"generation:.*,\s+([\d\.]+)\s+TPS", output)
    
    prefill_tps = float(prefill_match.group(2) if prefill_match and len(prefill_match.groups()) > 1 else prefill_match.group(1)) if prefill_match else 0.0
    gen_tps = float(gen_match.group(1)) if gen_match else 0.0
    
    return {
        "prefill_tps": prefill_tps,
        "gen_tps": gen_tps,
        "error": result.stderr if result.returncode != 0 else ""
    }


def calculate_statistics(values):
    """Calculate mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    
    return mean, std


def validate_model_path(model_path):
    """
    Validate model path to prevent command injection and path traversal.
    """
    # Normalize path to remove any '..' sequences
    try:
        normalized = os.path.normpath(model_path)
    except Exception as e:
        raise ValueError(f"Invalid path format: {e}")
    
    # Restrict models to the directory managed by build_android.sh --install.
    allowed_prefix = "/data/local/tmp/Quick.AI/"
    
    # Ensure path starts with allowed prefix
    if not normalized.startswith(allowed_prefix):
        raise ValueError(
            f"Model path must start with '{allowed_prefix}'. "
            f"Got: {model_path}"
        )
    
    # Validate characters: allow only safe filesystem characters
    # Allow: alphanumeric, hyphen, underscore, dot, forward slash, plus sign
    safe_chars_pattern = r'^[a-zA-Z0-9_\-./+]+$'
    if not re.match(safe_chars_pattern, normalized):
        raise ValueError(
            f"Model path contains invalid characters. "
            f"Only alphanumeric, '-', '_', '.', '/', and '+' are allowed. "
            f"Got: {model_path}"
        )
    
    # Prevent empty path segments (like double slashes)
    if '//' in normalized:
        raise ValueError(
            f"Model path contains empty segments (double slashes). "
            f"Got: {model_path}"
        )
    
    return normalized


def output_results_table(all_results, model_name, model_size, model_type, dtype, device):
    """Output all benchmark results in a pretty table format."""
    # Prepare table data
    headers = ["Threads", "Prompt", "Gen", "Prefill TPS", "Gen TPS"]
    table_data = []
    
    for result in all_results:
        prefill_str = f"{result['prefill_mean']:.2f} ± {result['prefill_std']:.2f}" if result['prefill_mean'] > 0 else "N/A"
        gen_str = f"{result['gen_mean']:.2f} ± {result['gen_std']:.2f}" if result['gen_mean'] > 0 else "N/A"
        
        table_data.append([
            result['n_threads'],
            result['n_prompt'],
            result['n_gen'],
            prefill_str,
            gen_str
        ])
    
    print("\n" + "=" * 90)
    print("BENCHMARK SWEEP RESULTS")
    print("=" * 90)
    print(f"Model: {model_name} | Size: {model_size} | Type: {model_type} | Dtype: {dtype} | Device: {device}")
    print("=" * 90)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 90)


def parse_list_arg(arg_string):
    """Parse comma-separated list argument."""
    if not arg_string:
        return []
    return [int(x.strip()) for x in arg_string.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="nntrainer benchmark with configuration sweeping for nntrainer CausalLM models"
    )
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model directory path (on device, e.g., /data/local/tmp/Quick.AI/models/qwen3-0.6b-q40)")
    parser.add_argument("-p", "--n-prompt", type=str, default="512",
                        help="Number of prompt tokens, comma-separated (default: 512)")
    parser.add_argument("-n", "--n-gen", type=str, default="0",
                        help="Number of generation tokens, comma-separated (default: 0)")
    parser.add_argument("-r", "--n-trials", type=int, default=5,
                        help="Number of trials per configuration (default: 5)")
    parser.add_argument("-t", "--n-threads", type=str, default="4",
                        help="Number of OMP threads, comma-separated (default: 4)")
    parser.add_argument("-b", "--batch-size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--device-info", type=str, default=None,
                        help="Device info (auto-detect if not specified)")
    parser.add_argument("--cool-to", type=float, default=35.0,
                        help="Cool device to this temperature before each config (default: 35.0)")
    parser.add_argument("--max-cool-wait", type=int, default=300,
                        help="Maximum wait time for cooling in seconds (default: 300)")
    parser.add_argument("--skip-cooling", action="store_true",
                        help="Skip cooling between configurations")
    
    args = parser.parse_args()
    
    # Parse list arguments
    n_prompts = parse_list_arg(args.n_prompt)
    n_gens = parse_list_arg(args.n_gen)
    n_threads_list = parse_list_arg(args.n_threads)
    
    for n_threads in n_threads_list:
        assert n_threads > 0, "Error: Thread counts must be positive integers"
    
    # Generate all configurations
    configs = list(product(n_prompts, n_gens, n_threads_list))
    
    # Extract model name from path
    model_path = validate_model_path(args.model)
    model_name = os.path.basename(model_path)
    
    print(f"=== nntrainer benchmark sweep ===")
    print(f"Model: {model_name}")
    print(f"Device path: {model_path}")
    print(f"n_prompt values: {n_prompts}")
    print(f"n_gen values: {n_gens}")
    print(f"n_threads values: {n_threads_list}")
    print(f"n_trials per config: {args.n_trials}")
    print(f"batch_size: {args.batch_size}")
    print(f"Total configurations: {len(configs)}")
    print("-" * 50)
    
    # Load nntr_config.json from device
    try:
        device_config_path = f"{model_path}/nntr_config.json"
        result = subprocess.run(
            ["adb", "shell", "cat", device_config_path],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Could not read nntr_config.json from device: {result.stderr}")
        
        nntr_cfg = json.loads(result.stdout)
        print("Successfully loaded nntr_config.json from device")
    except Exception as e: 
        print(f"Error loading nntr_config.json: {e}")
        return
    
    # Extract model metadata
    model_type = nntr_cfg.get("model_type", "Unknown")
    dtype = nntr_cfg.get("model_tensor_type", "Unknown")
    
    # Get model size
    model_size = get_model_size(model_path, nntr_cfg)
    print(f"Model size: {model_size}")
    print(f"Model type: {model_type}")
    print(f"Dtype: {dtype}")
    
    # Get device info
    device = args.device_info if args.device_info else get_device_model()
    print(f"Device: {device}")
    print("-" * 50)
    
    # Run benchmarks for all configurations
    all_results = []
    
    for idx, (n_prompt, n_gen, n_threads) in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Config: n_prompt={n_prompt}, n_gen={n_gen}, n_threads={n_threads}")
        print("-" * 50)
        
        # Wait for cooling before starting next configuration (for fair comparison)
        if idx > 1 and not args.skip_cooling:
            print("\nWaiting for device cooling...")
            wait_for_cooling(args.cool_to, args.max_cool_wait)
            time.sleep(2)  # Brief pause after cooling
        
        # Create config modifier for this specific configuration
        config_modifier = backup_and_modify_config(model_path, n_prompt, n_gen, args.batch_size)
        
        try:
            # Manually enter context to ensure proper cleanup on interrupt
            config_modifier.__enter__()
            
            results = []
            for i in range(args.n_trials):
                res = run_single_trial(model_path, n_threads)
                results.append(res)
                time.sleep(1)  # Brief pause between trials
            
            # Calculate statistics
            prefills = [r["prefill_tps"] for r in results if r["prefill_tps"] > 0]
            gens = [r["gen_tps"] for r in results if r["gen_tps"] > 0]
            
            prefill_mean, prefill_std = calculate_statistics(prefills)
            gen_mean, gen_std = calculate_statistics(gens)
            
            all_results.append({
                'n_prompt': n_prompt,
                'n_gen': n_gen,
                'n_threads': n_threads,
                'prefill_mean': prefill_mean,
                'prefill_std': prefill_std,
                'gen_mean': gen_mean,
                'gen_std': gen_std
            })
            
            print(f"  Prefill: {prefill_mean:.2f} ± {prefill_std:.2f} TPS")
            print(f"  Generation: {gen_mean:.2f} ± {gen_std:.2f} TPS")
            
            # Normal completion - restore config
            config_modifier.__exit__(None, None, None)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Restoring config...")
            # Config will be restored in __exit__ even on interrupt
            config_modifier.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
            break
        except Exception as e:
            print(f"Error in configuration: {e}")
            print("Restoring config...")
            # Config will be restored in __exit__ even on error
            config_modifier.__exit__(type(e), e, e.__traceback__)
            continue
    
    print("\n" + "=" * 50)
    print("ALL RESULTS")
    print("=" * 50)
    
    # Output all results in table format
    output_results_table(all_results, model_name, model_size, model_type, dtype, device)


if __name__ == "__main__":
    main()
