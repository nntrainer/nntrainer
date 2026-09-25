"""Mistral adapter. Sliding-window attention is read from
config.sliding_window at runtime by the shared base class -- listed
here only so `model_type == "mistral"` matches explicitly."""
from api.adapters.llama_family import LlamaFamilyAdapter


class MistralAdapter(LlamaFamilyAdapter):
    model_types = ("mistral",)
