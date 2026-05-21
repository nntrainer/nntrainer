"""Unified operation registry for mapping PyTorch ops to NNTrainer layer types.

This module centralizes the mapping tables that were previously duplicated
across _map_function_node() and _map_method_node() in NodeMapper.

Three categories of mappings:
  1. FUNCTION_OPS  - torch.* functions and operator.* to layer types
  2. METHOD_OPS    - Tensor.* method names to layer types
  3. Shared tables - ops that appear in both contexts (arithmetic, trig, etc.)
"""

import operator
import torch
import torch.nn.functional as F

from nntrainer_layers import (
    LAYER_ADDITION, LAYER_SUBTRACT, LAYER_MULTIPLY, LAYER_DIVIDE,
    LAYER_MATMUL, LAYER_POW, LAYER_SQRT, LAYER_NEGATIVE,
    LAYER_SIN, LAYER_COS, LAYER_TAN,
    LAYER_EXP, LAYER_LOG, LAYER_CLAMP,
    LAYER_REDUCE_MEAN, LAYER_REDUCE_SUM, LAYER_FLATTEN,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_CONCAT, LAYER_POOLING2D,
    LAYER_GATHER, LAYER_SLICE, LAYER_TOPK, LAYER_ARGSORT, LAYER_L2NORM,
    LAYER_LOSS_MSE, LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,
    LAYER_LOSS_CROSS_ENTROPY_SIGMOID, LAYER_LOSS_KLD,
    ACT_RELU, ACT_LEAKY_RELU, ACT_GELU, ACT_SWISH, ACT_SIGMOID, ACT_TANH, ACT_SOFTMAX,
    OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE, OP_SDPA, OP_NOOP, OP_UNSUPPORTED,
)


# ---------------------------------------------------------------------------
# Function-based ops: maps (callable) -> layer_type
# Simple ops that only need layer_type + standard input_layers.
# ---------------------------------------------------------------------------
FUNCTION_SIMPLE_OPS = {
    # Arithmetic
    torch.add:         LAYER_ADDITION,
    operator.add:      LAYER_ADDITION,
    torch.sub:         LAYER_SUBTRACT,
    operator.sub:      LAYER_SUBTRACT,
    torch.mul:         LAYER_MULTIPLY,
    operator.mul:      LAYER_MULTIPLY,
    torch.div:         LAYER_DIVIDE,
    operator.truediv:  LAYER_DIVIDE,
    torch.matmul:      LAYER_MATMUL,
    torch.pow:         LAYER_POW,
    torch.sqrt:        LAYER_SQRT,
    torch.neg:         LAYER_NEGATIVE,
    operator.neg:      LAYER_NEGATIVE,
    # Trigonometric
    torch.sin:         LAYER_SIN,
    torch.cos:         LAYER_COS,
    torch.tan:         LAYER_TAN,
    # Reduction
    torch.mean:        LAYER_REDUCE_MEAN,
    torch.sum:         LAYER_REDUCE_SUM,
    # Shape
    torch.flatten:     LAYER_FLATTEN,
    torch.reshape:     OP_RESHAPE,
    torch.transpose:   OP_TRANSPOSE,
    torch.outer:       LAYER_MATMUL,
}

# Function ops looked up by name (func.__name__) rather than identity.
# These are for cases where the callable can't be compared by `is`.
FUNCTION_NAME_SIMPLE_OPS = {
    "exp":  LAYER_EXP,
    "log":  LAYER_LOG,
    "pow":  LAYER_POW,
    "multiply": LAYER_MULTIPLY,  # torch.multiply (alias for torch.mul)
    "topk": LAYER_TOPK,
    "argsort": LAYER_ARGSORT,
    "outer": LAYER_MATMUL,  # torch.outer(a, b) ≈ matmul for 1D vectors
}

# Functions that map to OP_NOOP (internal torch/runtime functions)
FUNCTION_NOOP_NAMES = frozenset({
    "_set_grad_enabled", "tensor", "arange", "device",
    "zeros", "zeros_like", "ones", "ones_like",
    "full_like", "empty_like",
    "custom_function_call",
    # Comparison / conditional ops (T5 relative position bias)
    "where", "min", "max", "maximum", "minimum",
    # Debug / shape assertions (torchvision ViT, etc.)
    "_assert",
    # Position ID computation (XLM-RoBERTa, etc.)
    # cumsum is used exclusively for computing position IDs from attention
    # masks in transformer models. NNTrainer handles position IDs internally,
    # so the entire position ID chain (cumsum → arithmetic → embedding) is
    # redundant. The decomposer's _remove_position_id_chains() cleans up
    # the remaining arithmetic layers in the chain.
    "cumsum",
})

# Functions that map to OP_RESHAPE
FUNCTION_RESHAPE_NAMES = frozenset({"unsqueeze", "squeeze"})

# Functions that map to OP_PERMUTE
FUNCTION_PERMUTE_NAMES = frozenset({"permute", "roll"})

# Functions that map to OP_TRANSPOSE
FUNCTION_TRANSPOSE_NAMES = frozenset({"swapaxes", "swapdims", "movedim"})

# Functions that map to OP_UNSUPPORTED with decomposition info
FUNCTION_DECOMPOSE_OPS = {
    "rsqrt":      {"original_op": "rsqrt",      "decompose_to": "pow(x, -0.5)"},
    "abs":        {"original_op": "abs",         "decompose_to": "sqrt(pow(x, 2))"},
    "reciprocal": {"original_op": "reciprocal",  "decompose_to": "divide(1, x)"},
}

# Function-based activation mappings: callable -> activation type
FUNCTION_ACTIVATION_OPS = {
    F.silu:     ACT_SWISH,
    F.gelu:     ACT_GELU,
    F.relu:     ACT_RELU,
    F.leaky_relu: ACT_LEAKY_RELU,
    F.softmax:  ACT_SOFTMAX,
    torch.sigmoid: ACT_SIGMOID,
}

# Function activation by name (for torch.tanh / F.tanh ambiguity)
FUNCTION_ACTIVATION_NAMES = {
    "tanh": ACT_TANH,
    "leaky_relu": ACT_LEAKY_RELU,
    "log_softmax": ACT_SOFTMAX,  # log_softmax ≈ softmax for NNTrainer mapping
    "glu": ACT_SIGMOID,  # GLU(x) = x_a * sigmoid(x_b); map to sigmoid activation
    # Hard activation approximations: NNTrainer doesn't have native hard variants,
    # but the smooth counterparts are functionally equivalent for inference.
    "hardswish": ACT_SWISH,      # hardswish ≈ swish (x * sigmoid(x))
    "hardsigmoid": ACT_SIGMOID,  # hardsigmoid ≈ sigmoid
    "hardtanh": ACT_TANH,        # hardtanh(x) = clamp(x, -1, 1) ≈ tanh
}

# Function-based loss ops: callable -> layer_type
FUNCTION_LOSS_OPS = {
    F.cross_entropy:        LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,
    F.mse_loss:             LAYER_LOSS_MSE,
    F.kl_div:               LAYER_LOSS_KLD,
    F.binary_cross_entropy_with_logits: LAYER_LOSS_CROSS_ENTROPY_SIGMOID,
}

# Loss function name mappings (for name-based lookup)
FUNCTION_LOSS_NAMES = {
    "cross_entropy":        LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,
    "mse_loss":             LAYER_LOSS_MSE,
    "kl_div":               LAYER_LOSS_KLD,
    "binary_cross_entropy_with_logits": LAYER_LOSS_CROSS_ENTROPY_SIGMOID,
    "l1_loss":              LAYER_LOSS_MSE,  # L1 mapped to MSE as closest
    "nll_loss":             LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,  # NLL after log_softmax
}

# Function-based identity ops: callable -> layer_type
# These are checked by `func is <target>` and need special handling
# (e.g. multi-input extraction or parameter extraction).
FUNCTION_IDENTITY_OPS = {
    F.dropout:  LAYER_DROPOUT,
    F.pad:      OP_NOOP,
}

# Function names requiring special handling (not simple table lookups)
FUNCTION_INTERPOLATE_NAMES = frozenset({
    "interpolate",
})

FUNCTION_NORMALIZE_NAMES = frozenset({
    "normalize",
})

# Function-based clamp names (matching METHOD_CLAMP_NAMES pattern)
FUNCTION_CLAMP_NAMES = frozenset({
    "clamp", "clip", "clamp_min", "clamp_max",
})

# Function-based pooling names: maps func_name -> layer_type
FUNCTION_POOLING_NAMES = frozenset({
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d",
    "max_pool1d", "avg_pool1d",
})

# Layer types that produce tuple outputs (for operator.getitem handling)
MULTI_OUTPUT_LAYER_TYPES = frozenset({
    "gru", "lstm", "rnn", "lstmcell", "split",
})

# ---------------------------------------------------------------------------
# Method-based ops: maps method_name (str) -> layer_type
# Simple ops that only need layer_type + standard input_layers.
# ---------------------------------------------------------------------------
METHOD_SIMPLE_OPS = {
    # Arithmetic
    "add":    LAYER_ADDITION,
    "add_":   LAYER_ADDITION,
    "sub":    LAYER_SUBTRACT,
    "sub_":   LAYER_SUBTRACT,
    "mul":    LAYER_MULTIPLY,
    "mul_":   LAYER_MULTIPLY,
    "div":    LAYER_DIVIDE,
    "div_":   LAYER_DIVIDE,
    "__rdiv__":   LAYER_DIVIDE,    # reverse divide (scalar / tensor)
    "__rtruediv__": LAYER_DIVIDE,
    "__floordiv__": LAYER_DIVIDE,  # floor division (integer division)
    "matmul": LAYER_MATMUL,
    "neg":    LAYER_NEGATIVE,
    "pow":    LAYER_POW,
    "__rpow__":   LAYER_POW,       # reverse pow (scalar ** tensor)
    "sqrt":   LAYER_SQRT,
    "norm":   LAYER_L2NORM,     # tensor.norm() computes L2 norm
    # Trigonometric
    "cos":    LAYER_COS,
    "sin":    LAYER_SIN,
    # Reduction
    "mean":   LAYER_REDUCE_MEAN,
    "sum":    LAYER_REDUCE_SUM,
    # Shape
    "flatten": LAYER_FLATTEN,
    # Math
    "exp":    LAYER_EXP,
    "log":    LAYER_LOG,
    # Indexing
    "__getitem__": LAYER_SLICE,
    # Selection / sorting
    "topk":    LAYER_TOPK,
    "argsort": LAYER_ARGSORT,
}

# Method-based activation mappings: method_name -> activation type
METHOD_ACTIVATION_OPS = {
    "relu":      ACT_RELU,
    "relu_":     ACT_RELU,
    "sigmoid":   ACT_SIGMOID,
    "sigmoid_":  ACT_SIGMOID,
    "tanh":      ACT_TANH,
    "tanh_":     ACT_TANH,
    "softmax":   ACT_SOFTMAX,
}

# Method-based shape ops: method_name -> layer_type
METHOD_SHAPE_OPS = {
    "view":      OP_RESHAPE,
    "reshape":   OP_RESHAPE,
    "unflatten": OP_RESHAPE,
    "roll":      OP_PERMUTE,
    "swapaxes":  OP_TRANSPOSE,
    "swapdims":  OP_TRANSPOSE,
    "permute":   OP_PERMUTE,
    "transpose": OP_TRANSPOSE,
    "t":         OP_TRANSPOSE,    # .t() is 2D transpose
}

# Method ops that map to OP_RESHAPE
METHOD_RESHAPE_NAMES = frozenset({
    "unsqueeze", "squeeze", "repeat", "expand_as", "repeat_interleave",
    "as_strided",  # advanced view manipulation (Zipformer attention)
})

# Method ops that map to OP_UNSUPPORTED with decomposition info
METHOD_DECOMPOSE_OPS = {
    "rsqrt": {"original_op": "rsqrt", "decompose_to": "pow(x, -0.5)"},
    "abs":   {"original_op": "abs",   "decompose_to": "sqrt(pow(x, 2))"},
}

# Method ops that are clamp variants
METHOD_CLAMP_NAMES = frozenset({
    "clamp", "clip", "clamp_", "clamp_min", "clamp_max",
})

# Method ops that are no-ops in inference
METHOD_NOOP_NAMES = frozenset({
    "contiguous", "detach", "clone", "copy_", "to", "float",
    "half", "bfloat16", "int", "long", "short", "bool",
    "type_as", "expand",
    "size", "dim", "numel", "is_floating_point",
    "new_ones", "new_zeros", "new_full", "new_empty",
    "fill_", "zero_", "masked_fill", "masked_fill_",
    "__bool__", "__or__", "__and__",
    "__xor__", "__invert__", "__eq__", "__setitem__",
    "all", "any", "item",
    "stride",  # tensor metadata query (used in as_strided patterns)
    # Comparison ops (T5 relative position bias)
    "gt", "lt", "le", "ge", "eq", "ne",
    # Position ID computation (see FUNCTION_NOOP_NAMES comment for cumsum)
    "cumsum",
})

# Method ops that map to "split"
METHOD_SPLIT_NAMES = frozenset({"chunk", "split", "split_with_sizes", "unbind"})
