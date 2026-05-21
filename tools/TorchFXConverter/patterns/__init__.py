"""Pattern detection package for NNTrainer TorchFX converter.

Detects high-level structural patterns from flat layer lists:
  - AttentionPattern: Q/K/V/O projections, optional Q/K norms, SDPA, RoPE
  - FFNPattern: SwiGLU, GELU-FFN, standard FFN
  - TransformerBlockPattern: Norm + Attention + Residual + Norm + FFN + Residual
  - ModelStructure: Full model overview (embedding, blocks, LM head)

Sub-modules:
  - data_types.py  : Pattern dataclasses
  - config.py      : Config metadata extraction and arch inference
  - scope.py       : Block/attention/FFN scope discovery
  - attention.py   : Attention pattern detection
  - ffn.py         : FFN pattern detection
  - block.py       : Block-level detection (norms, residuals, operators)
"""

from nntrainer_layers import LAYER_RMS_NORM, LAYER_LAYER_NORM

from .data_types import (
    AttentionPattern, FFNPattern, SSMPattern, TransformerBlockPattern,
    ModelStructure, print_block,
)
from .config import extract_config_metadata, detect_embedding_and_head, infer_arch_type
from .scope import find_block_scopes
from .block import detect_block


class PatternDetector:
    """Detects structural patterns from a flat list of NNTrainerLayerDef.

    Detection is based on:
      1. Module hierarchy (hf_module_name)
      2. Layer types and connections (input_layers)
      3. Model config (optional)
    """

    def __init__(self, layers, model_config=None):
        self.layers = layers
        self.config = model_config
        self._by_name = {l.name: l for l in layers}
        self._idx_by_name = {l.name: i for i, l in enumerate(layers)}

    def detect(self):
        """Run full pattern detection pipeline.

        Returns:
            ModelStructure with all detected patterns.
        """
        structure = ModelStructure()

        # Step 1: Extract config metadata
        extract_config_metadata(structure, self.config)

        # Step 2: Detect embedding and LM head
        detect_embedding_and_head(structure, self.layers)

        # Step 3: Detect transformer blocks
        block_scopes = find_block_scopes(self.layers)

        encoder_scopes = [s for s in block_scopes if "encoder" in s]
        decoder_scopes = [s for s in block_scopes if "decoder" in s]
        other_scopes = [s for s in block_scopes
                        if "encoder" not in s and "decoder" not in s]

        enc_idx = 0
        for scope in encoder_scopes:
            block = detect_block(
                enc_idx, scope, self.layers, self.layers,
                self.config, self._by_name, self._idx_by_name)
            if block:
                block.block_role = "encoder"
                structure.blocks.append(block)
                enc_idx += 1
        structure.num_encoder_layers = enc_idx

        dec_idx = 0
        for scope in decoder_scopes:
            block = detect_block(
                dec_idx, scope, self.layers, self.layers,
                self.config, self._by_name, self._idx_by_name)
            if block:
                block.block_role = "decoder"
                structure.blocks.append(block)
                dec_idx += 1
        structure.num_decoder_layers = dec_idx

        for block_idx, scope in enumerate(other_scopes):
            block = detect_block(
                block_idx, scope, self.layers, self.layers,
                self.config, self._by_name, self._idx_by_name)
            if block:
                structure.blocks.append(block)

        structure.num_layers = len(structure.blocks)

        # Step 4: Final normalization
        self._detect_final_norm(structure)

        # Step 5: Infer architecture type
        infer_arch_type(
            structure, self.config,
            lambda: find_block_scopes(self.layers))

        return structure

    def _detect_final_norm(self, structure):
        """Detect the final normalization before LM head."""
        block_scopes = find_block_scopes(self.layers)
        for layer in reversed(self.layers):
            if layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
                if layer.hf_module_name and not any(
                    layer.hf_module_name.startswith(scope)
                    for scope in block_scopes
                    if scope + "." in layer.hf_module_name
                ):
                    structure.final_norm = layer.name
                    break


def detect_patterns(layers, model_config=None):
    """Detect structural patterns from a flat layer list.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        model_config: HuggingFace model config (optional).

    Returns:
        ModelStructure with all detected patterns.
    """
    detector = PatternDetector(layers, model_config)
    return detector.detect()
