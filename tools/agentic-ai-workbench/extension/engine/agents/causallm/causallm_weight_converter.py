"""
CausalLM Weight Converter Agent - Uses CausalLM's weight_converter.py scripts.

This agent calls the appropriate CausalLM weight converter script to generate
production-ready .bin files from HuggingFace model weights.
"""
import os
import sys
import subprocess
import shutil
from typing import Optional

from . import causallm_model_registry
from ..events import bus


def run(state: dict) -> dict:
    """
    Run the CausalLM weight converter.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with causallm_bin_path set
    """
    bus.agent_status("causallm_weight_converter", "running")

    if not state.get("weights_verified"):
        bus.log("Verified model weights are unavailable -- skipping CausalLM .bin generation", "error")
        bus.agent_status("causallm_weight_converter", "error", "weights not verified")
        return state
    
    # Get configuration from state or environment
    causallm_path = state.get("causallm_path") or os.environ.get("CAUSALLM_PATH")
    architecture = state.get("architecture")
    # The background downloader has already validated this local snapshot.
    # Passing it to the CausalLM converter prevents a second independent Hub
    # download and makes corruption gating effective.
    hf_model_path = state.get("hf_model_path") or state.get("weights_path") or state.get("model_name")
    out_dir = state.get("out_dir")
    
    # Auto-detect causallm_path if not provided
    if not causallm_path:
        causallm_path = causallm_model_registry.auto_detect_causallm_path()
        if causallm_path:
            bus.log(f"Auto-detected CausalLM path: {causallm_path}")
    
    if not causallm_path:
        bus.log("CausalLM path not configured -- skipping .bin generation", "warn")
        bus.agent_status("causallm_weight_converter", "done", "no causallm path")
        return state
    
    if not os.path.isdir(causallm_path):
        bus.log(f"CausalLM path does not exist: {causallm_path}", "warn")
        bus.agent_status("causallm_weight_converter", "error", "causallm path not found")
        return state
    
    if not architecture:
        bus.log("No architecture found in state -- cannot determine converter", "warn")
        bus.agent_status("causallm_weight_converter", "error", "no architecture")
        return state
    
    # A generated component owns its tensor map and converter. Falling back to
    # a handwritten architecture converter would silently restore handwritten
    # weight order, defeating generated-model parity.
    generated_converter = state.get("generated_weight_converter_path")
    generated_tensor_map = state.get("tensor_map_path")
    if generated_converter and generated_tensor_map:
        converter_path = generated_converter
        if not (os.path.isfile(converter_path) and os.path.isfile(generated_tensor_map)):
            bus.log("Generated tensor-map converter bundle is incomplete", "error")
            bus.agent_status("causallm_weight_converter", "error", "missing generated converter")
            return state
        use_generated_converter = True
    else:
        converter_path = causallm_model_registry.discover_converter(causallm_path, architecture)
        use_generated_converter = False
    if not converter_path:
        bus.log(f"No CausalLM converter found for architecture: {architecture}", "warn")
        bus.agent_status("causallm_weight_converter", "done", "no converter for this model")
        return state
    
    bus.log(f"Using CausalLM converter: {converter_path}")
    
    # Determine output path
    model_slug = architecture.replace("ForCausalLM", "").replace("Model", "").lower()
    output_bin = os.path.join(out_dir, f"nntr_{model_slug}_fp32.bin")
    
    # Check if output already exists (from cache)
    if os.path.exists(output_bin):
        size_mb = os.path.getsize(output_bin) / (1024 * 1024)
        bus.log(f"Using existing .bin file: {output_bin} ({size_mb:.1f} MB)")
        state["causallm_bin_path"] = output_bin
        state["causallm_bin_dtype"] = "FP32"
        state["weights_path"] = output_bin  # Also set weights_path for profiler
        bus.agent_status("causallm_weight_converter", "done", "cached")
        
        # Create model directory for profiler (even for cached files)
        try:
            from .causallm_so_builder import _create_model_directory
            model_dir = _create_model_directory(state)
            if model_dir:
                bus.log(f"Model directory ready: {model_dir}")
        except Exception as e:
            bus.log(f"Could not create model directory: {e}", "warn")
        return state
    
    # Build converter command
    # The converter script expects specific arguments
    if use_generated_converter:
        cmd = [
            sys.executable, converter_path,
            "--model-path", hf_model_path,
            "--tensor-map", generated_tensor_map,
            "--output-name", output_bin,
        ]
    else:
        cmd = [
            sys.executable, converter_path,
            "--model_path", hf_model_path,
            "--output_name", output_bin,
        ]
    
    # Add data type argument if specified
    output_dtype = state.get("output_dtype", "float32")
    if output_dtype:
        cmd.extend(["--data_type", output_dtype])
    
    bus.log(f"Running converter: {' '.join(cmd)}")
    
    try:
        # Run the converter
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(converter_path),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large models
        )
        
        if result.returncode != 0:
            bus.log(f"Converter failed: {result.stderr}", "error")
            bus.agent_status("causallm_weight_converter", "error", result.stderr[:200])
            return state
        
        # Log converter output
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    bus.log(f"  {line}")
        
        # Verify output
        if os.path.exists(output_bin):
            size_mb = os.path.getsize(output_bin) / (1024 * 1024)
            bus.log(f"Generated .bin file: {output_bin} ({size_mb:.1f} MB)")
            state["causallm_bin_path"] = output_bin
            state["causallm_bin_dtype"] = "FP32"
            state["weights_path"] = output_bin  # Also set weights_path for profiler
            bus.agent_status("causallm_weight_converter", "done", f"{size_mb:.1f} MB")
            
            # Create model directory for profiler
            try:
                from .causallm_so_builder import _create_model_directory
                model_dir = _create_model_directory(state)
                if model_dir:
                    bus.log(f"Model directory ready: {model_dir}")
            except Exception as e:
                bus.log(f"Could not create model directory: {e}", "warn")
        else:
            bus.log("Converter completed but .bin file not found", "error")
            bus.agent_status("causallm_weight_converter", "error", "output not found")
            
    except subprocess.TimeoutExpired:
        bus.log("Converter timed out (1 hour limit)", "error")
        bus.agent_status("causallm_weight_converter", "error", "timeout")
    except Exception as e:
        bus.log(f"Converter error: {e}", "error")
        bus.agent_status("causallm_weight_converter", "error", str(e))
    
    return state
