"""Qwen3 adapter. Structurally identical to LlamaFamilyAdapter's
generic handling (Q/K RMSNorm and GQA are both detected at runtime),
so this class exists purely to make `model_type == "qwen3"` an
explicit, visible match rather than falling through to the generic
Llama-family handling silently."""
from api.adapters.llama_family import LlamaFamilyAdapter


class Qwen3Adapter(LlamaFamilyAdapter):
    model_types = ("qwen3",)
