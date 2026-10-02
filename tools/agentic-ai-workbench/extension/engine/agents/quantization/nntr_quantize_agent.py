"""
NNTrainer Quantize Agent - Runs nntr_quantize for optional weight quantization.

This agent runs the nntr_quantize binary to convert FP32 .bin files to
quantized formats (Q4_0, Q6_K, etc.) for on-device deployment.
"""
import os
import subprocess
import shutil
from typing import Optional

from ..events import bus


# Quantization preset mappings
QUANTIZATION_PRESETS = {
    "none": {"fc_dtype": "FP32", "embd_dtype": "FP32", "lmhead_dtype": "FP32"},
    "Q4_0-FP32": {"fc_dtype": "Q4_0", "embd_dtype": "FP32", "lmhead_dtype": "FP32"},
    "Q6_K-Q4_0": {"fc_dtype": "Q4_0", "embd_dtype": "Q6_K", "lmhead_dtype": "Q4_0"},
    "Q4_0-Q6_K": {"fc_dtype": "Q4_0", "embd_dtype": "Q6_K", "lmhead_dtype": "Q6_K"},
}


def find_nntr_quantize(configured_path: Optional[str] = None) -> Optional[str]:
    """
    Find the nntr_quantize binary.
    
    Args:
        configured_path: Path from configuration
        
    Returns:
        Path to nntr_quantize binary if found, None otherwise
    """
    # Check configured path first
    if configured_path:
        if os.path.isfile(configured_path):
            return configured_path
        # Check if it's a directory, look for nntr_quantize inside
        bin_path = os.path.join(configured_path, "nntr_quantize")
        if os.path.isfile(bin_path):
            return bin_path
    
    # Search PATH
    nntr_quantize = shutil.which("nntr_quantize")
    if nntr_quantize:
        return nntr_quantize
    
    # Common locations
    search_paths = [
        "/usr/local/bin/nntr_quantize",
        "/usr/bin/nntr_quantize",
        os.path.expanduser("~/.local/bin/nntr_quantize"),
        os.path.expanduser("~/nntrainer/build/quantize"),
    ]
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    return None


def parse_preset(preset: str) -> dict:
    """
    Parse quantization preset into individual dtype settings.
    
    Args:
        preset: Preset name (e.g., "Q4_0-FP32")
        
    Returns:
        Dict with fc_dtype, embd_dtype, lmhead_dtype
    """
    if preset in QUANTIZATION_PRESETS:
        return QUANTIZATION_PRESETS[preset]
    
    # Try to parse custom preset (format: FC-EMBD or FC-EMBD-LMHEAD)
    parts = preset.upper().split("-")
    if len(parts) >= 2:
        return {
            "fc_dtype": parts[0],
            "embd_dtype": parts[1],
            "lmhead_dtype": parts[2] if len(parts) > 2 else parts[0]
        }
    
    # Default to FP32
    return {"fc_dtype": "FP32", "embd_dtype": "FP32", "lmhead_dtype": "FP32"}


def run(state: dict) -> dict:
    """
    Run nntr_quantize on the converted .bin file.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with quantized causallm_bin_path
    """
    bus.agent_status("nntr_quantize", "running")

    if not state.get("weights_verified"):
        bus.log("Weights are not verified -- skipping quantization", "error")
        bus.agent_status("nntr_quantize", "error", "weights not verified")
        return state
    
    # Check if quantization is enabled
    enable_quantization = state.get("enable_quantization", True)
    if not enable_quantization:
        bus.log("Quantization disabled in configuration -- skipping")
        bus.agent_status("nntr_quantize", "done", "disabled")
        return state
    
    # Get input .bin path
    bin_path = state.get("causallm_bin_path")
    if not bin_path:
        bus.log("No .bin file found -- skipping quantization", "warn")
        bus.agent_status("nntr_quantize", "done", "no input")
        return state
    
    if not os.path.exists(bin_path):
        bus.log(f".bin file not found: {bin_path}", "warn")
        bus.agent_status("nntr_quantize", "done", "input not found")
        return state
    
    # Find nntr_quantize binary
    nntr_quantize_path = state.get("nntr_quantize_path") or \
                         os.environ.get("NNTR_QUANTIZE_PATH")
    quantize_bin = find_nntr_quantize(nntr_quantize_path)
    
    if not quantize_bin:
        bus.log("nntr_quantize binary not found -- skipping quantization", "warn")
        bus.agent_status("nntr_quantize", "done", "binary not found")
        return state
    
    bus.log(f"Using nntr_quantize: {quantize_bin}")
    
    # Get quantization settings
    preset = state.get("quantization_preset", "Q4_0-FP32")
    target_isa = state.get("target_isa", "x86")
    dtypes = parse_preset(preset)
    
    bus.log(f"Quantization preset: {preset}")
    bus.log(f"  FC dtype: {dtypes['fc_dtype']}")
    bus.log(f"  Embedding dtype: {dtypes['embd_dtype']}")
    bus.log(f"  LM head dtype: {dtypes['lmhead_dtype']}")
    bus.log(f"  Target ISA: {target_isa}")
    
    # Get model directory (nntr_quantize expects a directory with config files)
    model_dir = os.path.dirname(bin_path)
    
    # Check for required config files
    config_path = os.path.join(model_dir, "config.json")
    nntr_config_path = os.path.join(model_dir, "nntr_config.json")
    
    if not os.path.exists(config_path):
        bus.log("config.json not found -- cannot quantize", "error")
        bus.agent_status("nntr_quantize", "error", "config.json missing")
        return state
    
    if not os.path.exists(nntr_config_path):
        bus.log("nntr_config.json not found -- creating minimal config", "warn")
        # Create minimal nntr_config.json
        import json
        minimal_config = {
            "model_type": "CausalLM",
            "model_file_name": os.path.basename(bin_path),
            "model_tensor_type": "FP32-FP32",
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32",
            "lmhead_dtype": "FP32"
        }
        with open(nntr_config_path, "w") as f:
            json.dump(minimal_config, f, indent=2)
    
    # Build quantization command
    cmd = [
        quantize_bin,
        model_dir,
        "--fc_dtype", dtypes["fc_dtype"],
        "--embd_dtype", dtypes["embd_dtype"],
        "--lmhead_dtype", dtypes["lmhead_dtype"],
        "--isa", target_isa,
    ]
    
    # Add output directory if different from input
    output_dir = state.get("out_dir")
    if output_dir and output_dir != model_dir:
        cmd.extend(["--output", output_dir])
    
    bus.log(f"Running quantization: {' '.join(cmd)}")
    
    try:
        # Run nntr_quantize
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large models
        )
        
        # Log output
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    bus.log(f"  {line}")
        
        if result.returncode != 0:
            bus.log(f"Quantization failed: {result.stderr}", "error")
            bus.agent_status("nntr_quantize", "error", result.stderr[:200])
            return state
        
        # Find output .bin file
        if output_dir and output_dir != model_dir:
            # Look for quantized file in output directory
            for f in os.listdir(output_dir):
                if f.endswith(".bin") and "quantized" in f.lower():
                    output_bin = os.path.join(output_dir, f)
                    break
            else:
                # Fallback: use same name as input but in output dir
                output_bin = os.path.join(output_dir, os.path.basename(bin_path).replace(".bin", f"_{dtypes['fc_dtype'].lower()}.bin"))
        else:
            # Output is in same directory - look for newly created file
            # nntr_quantize creates nntr_config_quantized.json, find the bin it references
            quantized_config = os.path.join(model_dir, "nntr_config_quantized.json")
            if os.path.exists(quantized_config):
                import json
                with open(quantized_config) as f:
                    config = json.load(f)
                output_bin = os.path.join(model_dir, config.get("model_file_name", ""))
            else:
                # Fallback naming
                output_bin = bin_path.replace(".bin", f"_{dtypes['fc_dtype'].lower()}.bin")
        
        # Verify output
        if os.path.exists(output_bin):
            orig_size = os.path.getsize(bin_path)
            new_size = os.path.getsize(output_bin)
            compression = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
            
            bus.log(f"Quantized .bin file: {output_bin}")
            bus.log(f"  Original size: {orig_size / (1024*1024):.1f} MB")
            bus.log(f"  Quantized size: {new_size / (1024*1024):.1f} MB")
            bus.log(f"  Compression: {compression:.1f}%")
            
            state["causallm_bin_path"] = output_bin
            state["causallm_bin_dtype"] = dtypes["fc_dtype"]
            state["causallm_bin_compression"] = f"{compression:.1f}%"
            bus.agent_status("nntr_quantize", "done", f"{compression:.1f}% compression")
        else:
            bus.log("Quantization completed but output file not found", "warn")
            bus.agent_status("nntr_quantize", "done", "output not verified")
            
    except subprocess.TimeoutExpired:
        bus.log("Quantization timed out (1 hour limit)", "error")
        bus.agent_status("nntr_quantize", "error", "timeout")
    except Exception as e:
        bus.log(f"Quantization error: {e}", "error")
        bus.agent_status("nntr_quantize", "error", str(e))
    
    return state
