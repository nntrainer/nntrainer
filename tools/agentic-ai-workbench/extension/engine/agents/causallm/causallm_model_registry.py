"""
CausalLM Model Registry - Maps HuggingFace architectures to CausalLM weight converters.

This module provides a registry of supported models and their corresponding
weight converter scripts from the CausalLM repository.
"""
import os
from typing import Optional, Dict, List
from pathlib import Path


# Mapping of HF architecture names to CausalLM converter paths
CAUSALLM_CONVERTER_MAP: Dict[str, str] = {
    # Qwen3 family
    "Qwen3ForCausalLM": "res/qwen3/qwen3-0.6b/weight_converter.py",
    "Qwen3MoeForCausalLM": "res/qwen3/qwen3-30b-a3b/weight_converter.py",
    "Qwen3SlimMoeForCausalLM": "res/qwen3/qwen3-30b-a3b-slim/weight_converter.py",
    "Qwen3CachedSlimMoeForCausalLM": "res/qwen3/qwen3-30b-a3b-slim-cached/weight_converter.py",
    
    # Gemma family
    "Gemma3ForCausalLM": "res/gemma3/weight_converter.py",
    "Gemma3TextModel": "res/gemma3/weight_converter.py",
    "Gemma4ForConditionalGeneration": "res/gemma4/weight_converter.py",
    
    # GPT-OSS family
    "GptOssForCausalLM": "res/gpt-oss/gpt-oss-20b/weight_converter.py",
    "GptOssCachedSlimCausalLM": "res/gpt-oss/gpt-oss-cached-slim/weight_converter.py",
    
    # Qwen2 family
    "Qwen2ForCausalLM": "res/qwen2/qwen2-0.5b/weight_converter.py",
    "Qwen2Model": "res/qwen2/qwen2-0.5b/weight_converter.py",
    
    # Llama family
    "LlamaForCausalLM": "res/llama/weight_converter.py",
    
    # Embedding models
    "Qwen3Embedding": "res/qwen3/qwen3-embedding/weight_converter.py",
    "Qwen2Embedding": "res/qwen2/qwen2-embedding/weight_converter.py",
    "EmbeddingGemma": "res/gemma-embedding/weight_converter.py",
    "MultilingualTinyBert": "res/tiny-bert/weight_converter.py",
    "DebertaV2": "res/deberta_v2/weight_converter.py",
    "XLMRobertaForMaskedLM": "res/xlm_roberta/weight_converter.py",
    
    # Vision models
    "TimmVisionTransformer": "res/vit/weight_converter.py",
    
    # Other
    "Lfm2ForCausalLM": "res/lfm2/weight_converter.py",
}

# Alternative architecture names that should map to the same converter
ARCHITECTURE_ALIASES: Dict[str, str] = {
    # Qwen3 variants
    "Qwen3": "Qwen3ForCausalLM",
    "Qwen3MoE": "Qwen3MoeForCausalLM",
    
    # Gemma variants  
    "Gemma3": "Gemma3ForCausalLM",
    "Gemma4": "Gemma4ForConditionalGeneration",
    
    # GPT-OSS variants
    "GptOss": "GptOssForCausalLM",
}


def get_supported_models() -> List[str]:
    """Return list of all supported architecture names."""
    return list(CAUSALLM_CONVERTER_MAP.keys())


def normalize_architecture(architecture: str) -> str:
    """
    Normalize architecture name by resolving aliases.
    
    Args:
        architecture: Raw architecture name from HF config
        
    Returns:
        Normalized architecture name
    """
    # First check if it's already a known architecture
    if architecture in CAUSALLM_CONVERTER_MAP:
        return architecture
    
    # Check aliases
    if architecture in ARCHITECTURE_ALIASES:
        return ARCHITECTURE_ALIASES[architecture]
    
    # Try removing common suffixes
    base = architecture
    for suffix in ["ForCausalLM", "ForSequenceClassification", "ForMaskedLM", 
                   "ForQuestionAnswering", "Model", "TextModel"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            if base in ARCHITECTURE_ALIASES:
                return ARCHITECTURE_ALIASES[base]
    
    return architecture


def discover_converter(causallm_root: str, architecture: str) -> Optional[str]:
    """
    Find the appropriate weight_converter.py for the given architecture.
    
    Args:
        causallm_root: Path to CausalLM repository root
        architecture: HuggingFace architecture name (e.g., "Qwen3ForCausalLM")
        
    Returns:
        Full path to weight_converter.py if found, None otherwise
    """
    if not causallm_root or not os.path.isdir(causallm_root):
        return None
    
    # Normalize architecture name
    normalized = normalize_architecture(architecture)
    
    # Look up converter path
    converter_rel = CAUSALLM_CONVERTER_MAP.get(normalized)
    if converter_rel:
        full_path = os.path.join(causallm_root, converter_rel)
        if os.path.exists(full_path):
            return full_path
    
    # Fallback: search for matching directories
    # Extract model slug from architecture (e.g., "Qwen3" from "Qwen3ForCausalLM")
    model_slug = normalized.replace("ForCausalLM", "").replace("Model", "").lower()
    
    # Search in res/ directory for matching converter
    res_dir = os.path.join(causallm_root, "res")
    if os.path.isdir(res_dir):
        for model_dir in os.listdir(res_dir):
            model_dir_lower = model_dir.lower()
            if model_slug in model_dir_lower or model_dir_lower in model_slug:
                # Check for weight_converter.py in this directory
                converter_path = os.path.join(res_dir, model_dir, "weight_converter.py")
                if os.path.exists(converter_path):
                    return converter_path
                # Check subdirectories
                for sub_dir in os.listdir(os.path.join(res_dir, model_dir)):
                    sub_path = os.path.join(res_dir, model_dir, sub_dir)
                    if os.path.isdir(sub_path):
                        converter_path = os.path.join(sub_path, "weight_converter.py")
                        if os.path.exists(converter_path):
                            return converter_path
    
    return None


def get_model_info(causallm_root: str) -> List[Dict]:
    """
    Get information about all available models in the CausalLM repository.
    
    Args:
        causallm_root: Path to CausalLM repository root
        
    Returns:
        List of dicts with model info: {architecture, converter_path, exists}
    """
    models = []
    for arch, converter_rel in CAUSALLM_CONVERTER_MAP.items():
        full_path = os.path.join(causallm_root, converter_rel) if causallm_root else None
        models.append({
            "architecture": arch,
            "converter_path": converter_rel,
            "exists": os.path.exists(full_path) if full_path else False
        })
    return models


def auto_detect_causallm_path() -> Optional[str]:
    """
    Auto-detect CausalLM repository path from common locations.
    
    Returns:
        Path to CausalLM repository if found, None otherwise
    """
    # Common locations to search
    search_paths = [
        # Relative to common workspace structures
        os.path.join(os.getcwd(), "Applications", "CausalLM"),
        os.path.join(os.getcwd(), "..", "Applications", "CausalLM"),
        os.path.join(os.getcwd(), "nntrainer", "Applications", "CausalLM"),
        
        # Home directory locations
        os.path.expanduser("~/nntrainer/Applications/CausalLM"),
        os.path.expanduser("~/projects/nntrainer/Applications/CausalLM"),
        
        # System-wide locations
        "/opt/nntrainer/Applications/CausalLM",
        "/usr/local/nntrainer/Applications/CausalLM",
    ]
    
    for path in search_paths:
        if os.path.isdir(path):
            # Verify it looks like a CausalLM directory
            if os.path.exists(os.path.join(path, "models")) or \
               os.path.exists(os.path.join(path, "res")):
                return path
    
    return None
