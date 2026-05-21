"""Backward-compatible facade for the patterns package.

All pattern detection logic now lives in the ``patterns/`` sub-package.
This module re-exports the public API so existing imports continue to work:

    from pattern_detector import PatternDetector, detect_patterns
    from pattern_detector import ModelStructure, TransformerBlockPattern
"""

from patterns import (                          # noqa: F401
    PatternDetector,
    detect_patterns,
    ModelStructure,
    TransformerBlockPattern,
    AttentionPattern,
    FFNPattern,
    SSMPattern,
    print_block as _print_block,
)
