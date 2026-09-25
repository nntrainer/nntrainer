"""
Architecture-specific "variant" C++ generators -- see registry.py for
why this exists. Importing this package registers every known variant;
callers should just do `from converters.variants import registry` and
call `registry.find_variant(architecture, hf_config)`.
"""
from . import qwen_moe
from .registry import find_variant, register_variant

register_variant("qwen3_moe", qwen_moe.detect, qwen_moe.generate)

__all__ = ["find_variant", "register_variant"]
