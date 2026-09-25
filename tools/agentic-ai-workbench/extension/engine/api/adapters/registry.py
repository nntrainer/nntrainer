"""
Adapter registry.

Plain list + linear scan is intentional: there are a handful of
adapters and `matches()` is a single attribute comparison, so this
never needs to be a dict keyed by model_type (which would silently
hide two adapters claiming the same model_type instead of surfacing
the conflict).

`register()` / `select_adapter()` are guarded by a lock so a plugin
adapter can be registered from a background thread (e.g. while the
weight-download thread is running) without racing the pipeline thread
that's simultaneously calling `select_adapter()`.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from api.adapters.base import ArchitectureAdapter
from api.adapters.llama import LlamaAdapter
from api.adapters.mistral import MistralAdapter
from api.adapters.qwen2 import Qwen2Adapter
from api.adapters.qwen3 import Qwen3Adapter

_lock = threading.Lock()

_ADAPTERS: list[ArchitectureAdapter] = [
    Qwen3Adapter(),
    Qwen2Adapter(),
    MistralAdapter(),
    LlamaAdapter(),
]


def register(adapter: ArchitectureAdapter) -> None:
    with _lock:
        _ADAPTERS.append(adapter)


def all_adapters() -> list[ArchitectureAdapter]:
    with _lock:
        return list(_ADAPTERS)


def select_adapter(config: Any, model: Any) -> Optional[ArchitectureAdapter]:
    """Returns the first adapter whose `matches()` is True, or None if
    this is a genuinely unknown architecture -- callers must treat
    None as "fall back to the module-tree view", never as an error."""
    for adapter in all_adapters():
        try:
            if adapter.matches(config, model):
                return adapter
        except Exception:
            # A single misbehaving adapter must never break selection
            # for every other model.
            continue
    return None


KNOWN_MODEL_TYPES: frozenset = frozenset(
    mt for adapter in all_adapters() for mt in adapter.model_types
)
