"""Qwen2 adapter (no Q/K norm, no sliding window by default -- both
still detected at runtime if a checkpoint enables them anyway)."""
from api.adapters.llama_family import LlamaFamilyAdapter


class Qwen2Adapter(LlamaFamilyAdapter):
    model_types = ("qwen2",)
