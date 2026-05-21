"""C++ registerCustomLayers() and initialize() method generation."""

from .helpers import _class_name


# Known NNTrainer layer type -> C++ class name mapping
# These must match the actual class names in CausalLM/layers/
CUSTOM_LAYER_CLASS = {
    "embedding_layer": "EmbeddingLayer",
    "tie_word_embeddings": "TieWordEmbedding",
    "rms_norm": "RMSNormLayer",
    "reshaped_rms_norm": "ReshapedRMSNormLayer",
    "mha_core": "MHACoreLayer",
    "swiglu": "SwiGLULayer",
    "short_conv": "ShortConvLayer",
}

# C++ class name -> header file mapping
CUSTOM_LAYER_HEADER = {
    "EmbeddingLayer": "embedding_layer.h",
    "TieWordEmbedding": "tie_word_embedding.h",
    "RMSNormLayer": "rms_norm.h",
    "ReshapedRMSNormLayer": "reshaped_rms_norm.h",
    "MHACoreLayer": "mha_core.h",
    "SwiGLULayer": "swiglu.h",
    "ShortConvLayer": "short_conv.h",
}


def collect_custom_layer_classes(structure, norm_type, attn_block):
    """Collect all custom C++ layer classes needed by this model.

    Args:
        structure: ModelStructure
        norm_type: "rms_norm" or "layer_normalization"
        attn_block: first block with attention, or None

    Returns:
        Sorted list of C++ class name strings.
    """
    s = structure
    classes = set()

    if s.embedding:
        classes.add("EmbeddingLayer")
    if s.tie_word_embeddings:
        classes.add("TieWordEmbedding")
    if norm_type == "rms_norm":
        classes.add("RMSNormLayer")

    if attn_block:
        classes.add("MHACoreLayer")
        if attn_block.attention.has_qk_norm:
            classes.add("ReshapedRMSNormLayer")

    for b in s.blocks:
        if b.ffn and b.ffn.ffn_type == "swiglu":
            classes.add("SwiGLULayer")
            break

    for b in s.blocks:
        for layer in (b.operator_layers or []):
            cls = CUSTOM_LAYER_CLASS.get(layer.layer_type)
            if cls:
                classes.add(cls)

    return sorted(classes)


def emit_custom_layer_includes(custom_classes):
    """Generate #include directives for custom layer headers.

    Args:
        custom_classes: sorted list of custom layer class names

    Returns:
        String of #include lines.
    """
    L = []
    for cls in custom_classes:
        header = CUSTOM_LAYER_HEADER.get(cls)
        if header:
            L.append(f"#include <{header}>")
    return "\n".join(L)


def emit_register_custom_layers(cname, custom_classes):
    """Generate registerCustomLayers() method body.

    Args:
        cname: C++ class name
        custom_classes: sorted list of custom layer class names
    """
    L = []

    L.append(f"void {cname}::registerCustomLayers() {{")
    L.append(f"  auto &ct_engine = nntrainer::Engine::Global();")
    L.append(f'  auto app_context =')
    L.append(f'    static_cast<nntrainer::AppContext *>('
             f'ct_engine.getRegisteredContext("cpu"));')
    L.append(f"")
    L.append(f"  try {{")

    for cls in custom_classes:
        L.append(f"    app_context->registerFactory("
                 f"nntrainer::createLayer<causallm::{cls}>);")

    L.append(f"  }} catch (std::invalid_argument &e) {{")
    L.append(f'    std::cerr << "failed to register factory, reason: " '
             f'<< e.what() << std::endl;')
    L.append(f"  }}")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def emit_allocate_kv_cache(cname):
    """Generate allocateKVCache() method body.

    Args:
        cname: C++ class name
    """
    L = []
    L.append(f"void {cname}::allocateKVCache() {{")
    L.append(f"  size_t max_timestep = INIT_SEQ_LEN + NUM_TO_GENERATE;")
    L.append(f"  size_t kv_heads = NUM_KV_HEADS;")
    L.append(f"  size_t cache_size = 1 * kv_heads * max_timestep * HEAD_DIM;")
    L.append(f"  kv_cache_buffers.allocate(NUM_LAYERS, cache_size);")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def emit_initialize(cname, external_kv_cache=False):
    """Generate initialize() method that wires up the full model pipeline.

    Calls registerCustomLayers() -> setProperty() -> constructModel()
    (which calls compile internally via symbolic graph) -> initialize.

    Args:
        cname: C++ class name
        external_kv_cache: whether to call allocateKVCache()
    """
    L = []
    L.append(f"void {cname}::initialize() {{")
    L.append(f"  // Set default sequence length if not configured")
    L.append(f"  if (INIT_SEQ_LEN == 0) {{")
    L.append(f"    INIT_SEQ_LEN = 8;")
    L.append(f"  }}")
    L.append(f"")
    L.append(f"  registerCustomLayers();")
    L.append(f"")
    L.append(f"  // constructModel() builds symbolic tensor graph,")
    L.append(f"  // compile and initialize are done internally by")
    L.append(f"  // model->compile(input, output, mode)")
    L.append(f"  constructModel();")
    if external_kv_cache:
        L.append(f"  allocateKVCache();")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)
