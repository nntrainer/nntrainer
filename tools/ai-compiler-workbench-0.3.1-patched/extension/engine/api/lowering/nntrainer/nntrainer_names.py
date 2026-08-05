"""
Translates source-framework (HuggingFace/torch) naming into the exact
identifiers nntrainer's C++ API expects. This has to happen at
*lowering* time, not at C++-emission time -- the target graph
(nntrainer_graph_ir) is supposed to be the authoritative "what will
nntrainer actually see" representation (see api/lowering/nntrainer),
so anything still holding a source-framework name (torch's "silu",
a plain python bool for `bias`) at that point is a lowering bug, not
a formatting detail for the C++ generator to paper over.
"""
from __future__ import annotations

from typing import Optional

#: torch/HF activation name -> nntrainer createLayer("activation", ...)
#: `activation=` property value. Deliberately explicit rather than a
#: passthrough default -- an activation nntrainer doesn't have should
#: fail loudly here, not silently emit a string nntrainer will reject.
NNTRAINER_ACTIVATION_NAMES = {
    "silu": "swish",
    "swish": "swish",
    "gelu": "gelu",
    "relu": "relu",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
    "mish": "mish",
    "elu": "elu",
    "selu": "selu",
    "softmax": "softmax",
}


def nntrainer_activation(source_activation: str) -> str:
    key = (source_activation or "").lower()
    if key not in NNTRAINER_ACTIVATION_NAMES:
        raise ValueError(
            f"No verified nntrainer activation for '{source_activation}' -- "
            "add it to NNTRAINER_ACTIVATION_NAMES once confirmed, rather than "
            "guessing a createLayer(\"activation\", ...) value nntrainer might reject."
        )
    return NNTRAINER_ACTIVATION_NAMES[key]


def disable_bias(has_bias: bool) -> str:
    """nntrainer's fully_connected layer takes `disable_bias`, not
    `bias` -- and it's a lowercase C++-style string, not a Python bool
    (str(False) == "False", which is not a value nntrainer's property
    parser recognizes)."""
    return "false" if has_bias else "true"


def bool_property(value: bool) -> str:
    return "true" if value else "false"


def reshaped_rms_norm_properties(epsilon: float, feature_size: int) -> dict:
    """Q/K-head RMSNorm (nntrainer's reshaped_rms_norm) needs `packed`
    in addition to epsilon/feature_size -- omitting it was flagged in
    review; nntrainer's own examples always pass it explicitly rather
    than relying on the layer's default."""
    return {"packed": "false", "epsilon": epsilon, "feature_size": feature_size}


#: architecture (CausalLMIR.architecture / HF config.model_type) -> the
#: generated Transformer/CausalLM class names a CausalLM-integration
#: project (e.g. nntrainer's own causallm example) would expect.
#: These are BEST-EFFORT names following the pattern the project's own
#: existing Qwen3 integration uses (Generated<Arch>Transformer /
#: Generated<Arch>CausalLM) -- verify against the real base-class
#: headers (transformer.h / causal_lm.h / <arch>_causallm.h) before
#: relying on them; a mismatched base-class name or method signature
#: will fail to compile, not silently produce wrong behavior, so this
#: is a compile-time-checkable assumption rather than a silent one.
TRANSFORMER_BASE_CLASSES = {
    "qwen3": "Qwen3Transformer",
    "qwen2": "Qwen2Transformer",
    "llama": "LlamaTransformer",
    "mistral": "MistralTransformer",
}

CAUSAL_LM_BASE_CLASSES = {
    "qwen3": "Qwen3CausalLM",
    "qwen2": "Qwen2CausalLM",
    "llama": "LlamaCausalLM",
    "mistral": "MistralCausalLM",
}


def generated_transformer_class(architecture: str) -> str:
    base = TRANSFORMER_BASE_CLASSES.get(architecture)
    prefix = "".join(part.capitalize() for part in architecture.split("_"))
    return f"Generated{base}" if base else f"Generated{prefix}Transformer"


def generated_causal_lm_class(architecture: str) -> str:
    base = CAUSAL_LM_BASE_CLASSES.get(architecture)
    prefix = "".join(part.capitalize() for part in architecture.split("_"))
    return f"Generated{base}" if base else f"Generated{prefix}CausalLM"
