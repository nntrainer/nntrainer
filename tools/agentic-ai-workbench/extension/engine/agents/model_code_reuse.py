"""
Model Code Reuse Agent - References existing implementations instead of generating.

For new models with similar architecture patterns, reference existing implementations
and only update class names. This avoids duplicating tested, working code.
"""
import os
import shutil
from pathlib import Path

from .events import bus

# Map of architecture patterns to reference implementations
REFERENCE_IMPLEMENTATIONS = {
    "Qwen3": "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/models/qwen3",
    "Qwen2": "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/models/qwen2",
    "Gemma3": "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/models/gemma3",
    "Gemma4": "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/models/gemma4",
    "Llama": "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM/models/qwen3",  # Use Qwen3 as reference
}


def find_reference_model(architecture: str) -> str:
    """Find a reference implementation for the given architecture."""
    slug = architecture.split("For")[0].replace("CausalLM", "")

    for pattern, path in REFERENCE_IMPLEMENTATIONS.items():
        if pattern.lower() in architecture.lower():
            return path

    # Default to Qwen3 if no match
    return REFERENCE_IMPLEMENTATIONS.get("Qwen3")


def adapt_implementation(ref_path: str, target_arch: str, output_dir: str) -> dict:
    """
    Adapt reference implementation to new architecture.
    Only change class names, keep all method implementations.

    LAYER REUSE STRATEGY:
    This adapted code reuses tested layer implementations from:
    /Applications/CausalLM/layers/

    Examples:
    - createLayer("reshaped_rms_norm", ...) → uses reshaped_rms_norm.cpp
    - createLayer("mha_core", ...) → uses mha_core.cpp
    - createLayer("fully_connected", ...) → uses nntrainer's built-in
    - registerCustomLayers() → registers ReshapedRMSNormLayer from layers/

    NO LAYER CODE IS DUPLICATED - only model architecture (class methods).
    """
    ref_path = Path(ref_path)
    output_path = Path(output_dir)

    # Find source files
    cpp_files = list(ref_path.glob("*_causallm.cpp"))
    h_files = list(ref_path.glob("*_causallm.h"))

    if not cpp_files or not h_files:
        bus.log(f"No reference implementation found in {ref_path}", "error")
        return {"success": False}

    ref_cpp = cpp_files[0]
    ref_h = h_files[0]

    # Read reference files
    with open(ref_cpp, 'r') as f:
        cpp_content = f.read()
    with open(ref_h, 'r') as f:
        h_content = f.read()

    # Extract reference model name from file
    ref_model_name = ref_h.stem.split("_")[0]  # e.g., "qwen3" from "qwen3_causallm.h"

    # Generate target model name
    target_slug = target_arch.split("For")[0].replace("CausalLM", "")  # e.g., "Llama" from "LlamaForCausalLM"
    target_model_lower = target_slug.lower()  # e.g., "llama"

    # Replace class names and includes (case-sensitive)
    replacements = [
        # Class names (Qwen3 -> Llama)
        (f"{ref_model_name.capitalize()}Transformer", f"{target_slug}Transformer"),
        (f"{ref_model_name.capitalize()}CausalLM", f"{target_slug}CausalLM"),
        # File names in includes
        (f"<{ref_model_name}_causallm.h>", f'"{target_slug.lower()}_causallm.h"'),
        (f'"{ref_model_name}_causallm.h"', f'"{target_slug.lower()}_causallm.h"'),
        # File comments
        (f"qwen3 causal", f"{target_slug.lower()} causal"),
        (ref_model_name.upper(), target_slug.upper()),
    ]

    cpp_content_new = cpp_content
    h_content_new = h_content

    for old, new in replacements:
        cpp_content_new = cpp_content_new.replace(old, new)
        h_content_new = h_content_new.replace(old, new)

    # Update header guards
    old_guard = f"__{ref_model_name.upper()}_CAUSAL_LM_H__"
    new_guard = f"__{target_slug.upper()}_CAUSAL_LM_H__"
    h_content_new = h_content_new.replace(old_guard, new_guard)
    h_content_new = h_content_new.replace(f"#define {old_guard}", f"#define {new_guard}")
    h_content_new = h_content_new.replace(f"#endif /* {old_guard}", f"#endif /* {new_guard}")

    # Write output files
    output_path.mkdir(parents=True, exist_ok=True)

    output_h = output_path / f"{target_model_lower}_causallm.h"
    output_cpp = output_path / f"{target_model_lower}_causallm.cpp"

    # Add layer reuse header comment to both files
    layer_reuse_notice = f"""
/*
 * LAYER REUSE NOTICE:
 * This {target_slug} implementation reuses tested layer implementations from:
 * /Applications/CausalLM/layers/
 *
 * Layer implementations used (NOT duplicated):
 * - reshaped_rms_norm (reshaped_rms_norm.cpp)
 * - mha_core (mha_core.cpp)
 * - fully_connected (nntrainer built-in)
 * - addition (nntrainer built-in)
 * - embedding (embedding_layer.cpp if needed)
 *
 * ONLY model architecture code (class methods) is in this file.
 * All layer implementations are referenced from CausalLM/layers/ folder.
 */
"""

    with open(output_h, 'w') as f:
        f.write(layer_reuse_notice)
        f.write(h_content_new)
    with open(output_cpp, 'w') as f:
        f.write(layer_reuse_notice)
        f.write(cpp_content_new)

    bus.log(f"Adapted {ref_model_name} → {target_slug}")
    bus.log(f"  Header: {output_h.name}")
    bus.log(f"  Source: {output_cpp.name}")
    bus.log(f"  ✅ Layer implementations reused from /layers/ (not duplicated)")

    return {
        "success": True,
        "header_path": str(output_h),
        "source_path": str(output_cpp),
        "reference_model": ref_model_name,
        "target_model": target_slug,
    }
