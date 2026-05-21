"""C++ code generation helpers and model naming utilities."""

import re


# =============================================================================
# C++ code helpers
# =============================================================================

def _q(s):
    """Quote a string as a C++ string literal."""
    return f'"{s}"'


def _cpp_layer(layer_type, props, indent=2):
    """Generate a createLayer() call as a list of C++ lines (old addLayer pattern)."""
    pad = "  " * indent
    lines = []
    lines.append(pad + 'layers.push_back(createLayer("' + layer_type + '", {')
    for i, p in enumerate(props):
        comma = "," if i < len(props) - 1 else ""
        lines.append(pad + "  " + p + comma)
    lines.append(pad + "}));")
    return lines


def _cpp_create_layer(layer_type, props, indent=1):
    """Generate a createLayer() expression (no push_back, no variable assignment).

    Returns a list of C++ lines forming: createLayer("type", {props...})
    Caller is responsible for wrapping with LayerHandle var(...) or assignment.
    """
    pad = "  " * indent
    lines = []
    lines.append(pad + 'createLayer("' + layer_type + '", {')
    for i, p in enumerate(props):
        comma = "," if i < len(props) - 1 else ""
        lines.append(pad + "  " + p + comma)
    lines.append(pad + "})")
    return lines


def _cpp_tensor_layer(var_name, layer_type, props, input_expr, indent=1):
    """Generate symbolic Tensor graph pattern:

      LayerHandle var(createLayer("type", {props...}));
      Tensor out = var(input);

    Args:
        var_name: C++ variable name for the LayerHandle
        layer_type: NNTrainer layer type string
        props: list of property expressions (without input_layers)
        input_expr: C++ expression for input - either a single Tensor variable
                    name (str) or a list/set expression like "{q, k, v}"

    Returns:
        tuple (lines, output_var) where output_var is the Tensor variable name.
    """
    pad = "  " * indent
    out_var = var_name + "_out"
    lines = []

    # LayerHandle declaration
    if len(props) <= 2:
        prop_str = ", ".join(props)
        lines.append(
            f'{pad}LayerHandle {var_name}(createLayer("{layer_type}", '
            f'{{{prop_str}}}));')
    else:
        lines.append(
            f'{pad}LayerHandle {var_name}(createLayer("{layer_type}", {{')
        for i, p in enumerate(props):
            comma = "," if i < len(props) - 1 else ""
            lines.append(f'{pad}  {p}{comma}')
        lines.append(f'{pad}}}));')

    # Tensor output
    lines.append(f'{pad}Tensor {out_var} = {var_name}({input_expr});')
    return lines, out_var


def _with_key(key, val):
    """Generate withKey() call as a C++ expression string."""
    return f'withKey("{key}", {val})'


# =============================================================================
# Model name helpers
# =============================================================================

_CLASS_NAME_MAP = {
    "qwen3": "Qwen3CausalLM",
    "qwen2": "Qwen2CausalLM",
    "llama": "LlamaCausalLM",
    "mistral": "MistralCausalLM",
    "gemma": "GemmaCausalLM",
    "gemma2": "Gemma2CausalLM",
    "gemma3_text": "Gemma3CausalLM",
    "phi": "PhiCausalLM",
    "bert": "BertModel",
    "roberta": "RobertaModel",
    "xlm-roberta": "XlmRobertaModel",
    "t5": "T5Model",
    "mt5": "MT5Model",
    "lfm2": "Lfm2CausalLM",
    "granitemoehybrid": "GraniteCausalLM",
}

_EMBED_NAME_MAP = {
    "gemma": "GemmaEmbeddingModel",
    "gemma2": "Gemma2EmbeddingModel",
    "gemma3_text": "Gemma3EmbeddingModel",
    "llama": "LlamaEmbeddingModel",
    "qwen3": "Qwen3EmbeddingModel",
    "qwen2": "Qwen2EmbeddingModel",
    "bert": "BertEmbeddingModel",
    "xlm-roberta": "XlmRobertaEmbeddingModel",
}

_KNOWN_CAUSAL_TYPES = frozenset({
    "qwen3", "qwen2", "llama", "mistral", "gemma",
    "gemma2", "gemma3_text", "phi", "gpt2",
    "gpt_neo", "gpt_neox", "starcoder2", "codegen",
    "lfm2", "granitemoehybrid",
})


def _class_name(model_type, arch_type):
    """Generate C++ class name from model type."""
    if arch_type == "embedding":
        if model_type in _EMBED_NAME_MAP:
            return _EMBED_NAME_MAP[model_type]
        return model_type.capitalize() + "EmbeddingModel"

    if model_type in _CLASS_NAME_MAP:
        return _CLASS_NAME_MAP[model_type]

    suffix = {"decoder_only": "CausalLM", "encoder_only": "Model",
              "encoder_decoder": "Model"}
    if model_type not in _KNOWN_CAUSAL_TYPES:
        return model_type.capitalize() + "Model"
    return model_type.capitalize() + suffix.get(arch_type, "Model")


def _file_basename(class_name):
    """Generate file basename from C++ class name (snake_case, no extension).

    e.g. "Qwen3CausalLM" -> "qwen3_causallm"
         "GemmaEmbeddingModel" -> "gemma_embedding_model"
    """
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', class_name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def _header_guard(class_name):
    """Generate header guard macro from class name."""
    basename = _file_basename(class_name).upper()
    return f"__{basename}_H__"


def _sanitize_model_name(model_name):
    """Sanitize a model name into a file-safe snake_case basename.

    e.g. "KaLM-embedding-v2.5" -> "kalm_embedding_v2_5"
         "Qwen/Qwen3-0.6B"     -> "qwen3_0_6b"
    """
    name = model_name.rstrip("/")
    name = name.rsplit("/", 1)[-1]
    name = re.sub(r'[-.\s]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name.lower()


def get_output_filenames(model_type, arch_type, model_name=None):
    """Get output filenames for a given model.

    Returns:
        dict with keys: "header", "source", "ini", "json"
    """
    if model_name:
        base = _sanitize_model_name(model_name)
    else:
        cname = _class_name(model_type, arch_type)
        base = _file_basename(cname)
    return {
        "header": f"{base}.h",
        "source": f"{base}.cpp",
        "ini": f"{base}.ini",
        "json": f"{base}.json",
    }


def get_file_base(structure, model_name=None):
    """Return the file basename, using model_name if provided."""
    if model_name:
        return _sanitize_model_name(model_name)
    cname = _class_name(structure.model_type, structure.arch_type)
    return _file_basename(cname)


def get_norm_type(model_type):
    """Return the norm layer type for a given model type."""
    layer_norm_types = ("bert", "roberta", "xlm-roberta")
    return ("layer_normalization" if model_type in layer_norm_types
            else "rms_norm")
