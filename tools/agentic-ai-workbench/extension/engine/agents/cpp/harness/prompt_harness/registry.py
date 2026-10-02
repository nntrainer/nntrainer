"""
Model registry for the Prompt Harness.

Tracks which models have prompt files, where they came from (reference
implementation vs LLM-generated), and whether they've been validated.

This is a NEW, additive registry -- it does not replace or modify the
existing hardcoded routing in llm_codegen.py's _get_prompt_module().
"""
import os
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict, field

_REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model_configs")
_REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "models.yaml")


@dataclass
class ModelEntry:
    """A single model's registration record"""
    name: str
    prompt_file: str
    source: str  # 'reference_implementation' | 'hf_llm_generated' | 'manual'
    status: str = "pending"  # 'pending' | 'ready' | 'failed'
    hf_id: Optional[str] = None
    reference_cpp: Optional[str] = None
    validated: bool = False
    validation_errors: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelRegistry:
    """
    Manages model_configs/models.yaml -- the central place to discover
    which models are registered, how their prompts were generated, and
    whether they've passed validation.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or _REGISTRY_FILE
        self._entries: Dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load existing registry, if present. Missing file = empty registry."""
        if not os.path.exists(self.registry_path):
            return

        try:
            import yaml
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            # Fallback: try JSON if yaml isn't available
            with open(self.registry_path, 'r') as f:
                data = json.load(f) or {}

        for name, entry_data in data.get('models', {}).items():
            self._entries[name] = ModelEntry(name=name, **entry_data)

    def save(self) -> None:
        """Persist registry to disk"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

        data = {
            'models': {
                name: {k: v for k, v in entry.to_dict().items() if k != 'name'}
                for name, entry in self._entries.items()
            }
        }

        try:
            import yaml
            with open(self.registry_path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)
        except ImportError:
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

    def get(self, name: str) -> Optional[ModelEntry]:
        return self._entries.get(name)

    def has(self, name: str) -> bool:
        return name in self._entries

    def register(self, entry: ModelEntry) -> None:
        self._entries[entry.name] = entry
        self.save()

    def list_all(self) -> Dict[str, ModelEntry]:
        return dict(self._entries)

    def list_by_status(self, status: str) -> Dict[str, ModelEntry]:
        return {n: e for n, e in self._entries.items() if e.status == status}
