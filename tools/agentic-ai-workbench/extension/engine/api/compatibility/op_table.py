"""
Op-level -> nntrainer mapping table.

Every op the parser hands us gets looked up here, once. Nothing in this
file knows what "Gemma" or "Llama" is: if a real HF model produces an op
that isn't listed, that's a genuine gap to see and fix, not something to
special-case per model.

Two hard rules keep this table honest:
  * Never emit a `createLayer("...")` type string we haven't verified
    exists in nntrainer. A recognized-but-unverified op is marked
    supported=False with a specific reason, NOT mapped to a guessed type
    (a wrong type string produces confidently-broken C++ -- exactly what
    this pipeline exists to avoid).
  * Unmapped/unknown ops still emit a graph node (emits_layer=True) so the
    gap is visible, never silently dropped.

NOTE ON THE ACTIVE PARSER: the current parser (GenericFxParser) is a
module-tree walker -- it only ever looks up leaf-module *class names*
(type(module).__name__). The ELEMENTWISE_OPS / PASSTHROUGH_OPS /
UNMAPPED_COMPUTE_OPS categories below are keyed by torch.fx
call_function/call_method names and are therefore dormant under the
module walker; they are kept for the fx path and for when dataflow-level
tracing (see ENHANCEMENTS item A) starts surfacing functional ops.
"""

# call_module class names -> a real nntrainer layer, 1:1
LAYER_OPS = {
    "Linear":      {"nntrainer_type": "fully_connected",     "supported": True, "reason": ""},
    # transformers' GPT-2 Conv1D is a linear projection (weight stored
    # transposed), NOT a convolution -- it maps cleanly to fully_connected.
    # (nn.Conv1d, a real 1-D convolution, has class name "Conv1d" and is
    # deliberately NOT matched here.)
    "Conv1D":      {"nntrainer_type": "fully_connected",     "supported": True,
                    "reason": "GPT-2 Conv1D is a linear projection -> fully_connected"},
    "Embedding":   {"nntrainer_type": "embedding",           "supported": True, "reason": ""},
    "LayerNorm":   {"nntrainer_type": "layer_normalization", "supported": True, "reason": ""},
    "Conv2d":      {"nntrainer_type": "conv2d",              "supported": True, "reason": ""},
    "BatchNorm2d": {"nntrainer_type": "batch_normalization", "supported": True, "reason": ""},
    "Dropout": {
        "nntrainer_type": None,
        "supported": True,
        "emits_layer": False,
        "reason": "no-op at inference, dropped rather than translated",
    },
}

# call_module / call_function names that are all "an activation" to nntrainer.
# Values must be activation strings nntrainer actually supports
# (relu, gelu, swish, tanh, sigmoid, softmax, mish, elu, selu).
ACTIVATION_OPS = {
    # torch nn modules
    "GELU": "gelu", "gelu": "gelu",
    "SiLU": "swish", "silu": "swish",
    "ReLU": "relu", "relu": "relu",
    "Tanh": "tanh", "tanh": "tanh",
    "Sigmoid": "sigmoid", "sigmoid": "sigmoid",
    "Mish": "mish", "mish": "mish",
    "ELU": "elu", "SELU": "selu",
    "softmax": "softmax",
    # HuggingFace transformers.activations module classes (BERT/GPT/etc.
    # use these wrappers rather than the bare torch nn modules)
    "GELUActivation": "gelu",
    "NewGELUActivation": "gelu",
    "FastGELUActivation": "gelu",
    "PytorchGELUTanh": "gelu",
    "SiLUActivation": "swish",
    "MishActivation": "mish",
    # NOTE: QuickGELUActivation / ClippedGELUActivation / LeakyReLU are
    # intentionally omitted -- they have no exact nntrainer equivalent, so
    # they fall through to "unknown" rather than being silently approximated.
}

# genuinely maps to an nntrainer layer (residual add)
ELEMENTWISE_OPS = {
    "add":  {"nntrainer_type": "addition", "supported": True, "reason": ""},
    "iadd": {"nntrainer_type": "addition", "supported": True, "reason": ""},
}

# shape/indexing bookkeeping only, no compute -> no layer emitted.
PASSTHROUGH_OPS = {
    "view", "reshape", "contiguous", "to", "size", "getitem",
    "transpose", "permute", "expand", "flatten", "unsqueeze", "squeeze",
}

# real compute, no primitive-level nntrainer equivalent in this table yet.
UNMAPPED_COMPUTE_OPS = {
    "matmul", "bmm", "mul", "sub", "div", "cat", "mean", "pow",
    "masked_fill", "where", "scaled_dot_product_attention",
}

# Recognized module families that nntrainer likely supports but whose exact
# createLayer type string we have NOT verified, so we refuse to emit a guessed
# name. Matched by class-name *suffix* (case-insensitive) because HF prefixes
# these with the model name: LlamaRMSNorm, Qwen2RMSNorm, GemmaRMSNorm, ...
# emits_layer=True so the node is visible; supported=False so it stays a
# clearly-explained manual step instead of a mysterious "unknown operator".
MANUAL_SUFFIX_OPS = {
    "RMSNorm": "RMS normalization -- nntrainer provides an RMS-norm layer; set "
               "the createLayer type by hand (not auto-emitted to avoid an "
               "unverified layer-type string)",
}


def _activation_entry(op_key):
    return {
        "nntrainer_type": "activation",
        "supported": True,
        "emits_layer": True,
        "reason": "",
        "attributes": {"activation": ACTIVATION_OPS[op_key]},
    }


def classify_op(op_key: str) -> dict:
    """
    Returns a dict describing how (or whether) a traced op maps to nntrainer:
        nntrainer_type: str | None
        supported:      bool
        emits_layer:    bool
        reason:         str
        attributes:     dict
    """
    if op_key in LAYER_OPS:
        entry = dict(LAYER_OPS[op_key])
        entry.setdefault("emits_layer", entry.get("nntrainer_type") is not None)
        entry.setdefault("attributes", {})
        return entry

    if op_key in ACTIVATION_OPS:
        return _activation_entry(op_key)

    if op_key in ELEMENTWISE_OPS:
        entry = dict(ELEMENTWISE_OPS[op_key])
        entry["emits_layer"] = True
        entry.setdefault("attributes", {})
        return entry

    if op_key in PASSTHROUGH_OPS:
        return {
            "nntrainer_type": None,
            "supported": True,
            "emits_layer": False,
            "reason": "shape/indexing op, passed through with no layer emitted",
            "attributes": {},
        }

    if op_key in UNMAPPED_COMPUTE_OPS:
        return {
            "nntrainer_type": None,
            "supported": False,
            "emits_layer": True,
            "reason": f"'{op_key}' has no primitive-level nntrainer mapping yet",
            "attributes": {},
        }

    # Suffix match for model-prefixed families (e.g. LlamaRMSNorm).
    for suffix, reason in MANUAL_SUFFIX_OPS.items():
        if op_key.endswith(suffix):
            return {
                "nntrainer_type": None,
                "supported": False,
                "emits_layer": True,
                "reason": reason,
                "attributes": {},
            }

    return {
        "nntrainer_type": None,
        "supported": False,
        "emits_layer": True,
        "reason": f"unknown operator '{op_key}' -- not in the op table",
        "attributes": {},
    }
