"""
Generator registry for C++ code generation harness.
"""

from typing import Dict, Optional
from .base import CppGeneratorHarness

_registry: Dict[str, CppGeneratorHarness] = {}


def register(harness: CppGeneratorHarness) -> None:
    """Register a generator harness."""
    _registry[harness.get_generator_id()] = harness


def get_generator(generator_id: str) -> Optional[CppGeneratorHarness]:
    """Get a generator by ID."""
    return _registry.get(generator_id)


def list_generators() -> Dict[str, list]:
    """List all registered generators with their capabilities."""
    return {gid: h.get_capabilities() for gid, h in _registry.items()}
