"""
C++ Generator Harness - Abstract base classes and utilities for C++ code generation.
"""

from .base import CppGeneratorHarness, GeneratedFiles
from .layer_catalog import LAYER_CATALOG, get_layer_info, get_available_layers, validate_layer_usage

__all__ = [
    'CppGeneratorHarness',
    'GeneratedFiles',
    'LAYER_CATALOG',
    'get_layer_info',
    'get_available_layers',
    'validate_layer_usage'
]
