"""Llama / Llama-2 / Llama-3 adapter."""
from api.adapters.llama_family import LlamaFamilyAdapter


class LlamaAdapter(LlamaFamilyAdapter):
    model_types = ("llama",)
