"""Legacy C++ emitter that generates code in the older addLayer pattern.

This emitter generates C++ code that uses the older pattern with:
- std::vector<LayerHandle> layers
- layers.push_back(createLayer(...))
- model->addLayer(layer) calls

Instead of the newer tensor flow pattern with Tensor objects.
"""

from .helpers import _class_name, get_file_base, get_norm_type


def emit_legacy_construct_model(structure, block_type, is_hybrid, blocks_info):
    """Generate constructModel() method using the legacy addLayer pattern."""
    s = structure
    cname = _class_name(s.model_type, s.arch_type)
    norm_type = get_norm_type(s.model_type)
    
    L = []
    L.append(f"void {cname}::constructModel() {{")
    L.append(f"  // Layers used in the model")
    L.append(f"  std::vector<LayerHandle> layers;")
    L.append(f"")
    L.append(f"  // Create model")
    L.append(f"  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);")
    L.append(f"")
    L.append(f"  // Set model properties")
    L.append(f"  std::vector<std::string> model_props = {{")
    L.append(f'    withKey("batch_size", 1),')
    L.append(f'    withKey("epochs", "1"),')
    L.append(f'    withKey("model_tensor_type", "FP32-FP32")')
    L.append(f"  }};")
    L.append(f"  model->setProperty(model_props);")
    L.append(f"")

    # Input layer
    L.append(f"  // Create input layer")
    L.append(f'  layers.push_back(createLayer("input", {{')
    L.append(f'    withKey("name", "input0"),')
    L.append(f'    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))')
    L.append(f"  }}));")
    L.append(f"")

    # Embedding
    if s.embedding:
        _emit_legacy_embedding(L, s)

    # Transformer blocks
    if s.arch_type == "encoder_decoder":
        _emit_legacy_encoder_decoder_blocks(L, s, norm_type)
    elif is_hybrid:
        _emit_legacy_hybrid_blocks(L, s, block_type, blocks_info)
    else:
        _emit_legacy_standard_blocks(L, s, block_type)

    # Final norm
    if s.arch_type != "encoder_decoder" and s.final_norm:
        _emit_legacy_final_norm(L, s, norm_type)

    # LM head
    if s.lm_head:
        _emit_legacy_lm_head(L, s)

    # Add all layers to the model
    L.append(f"  // Add all layers to the model")
    L.append(f"  for (auto &layer : layers) {{")
    L.append(f"    model->addLayer(layer);")
    L.append(f"  }}")
    L.append(f"}}")
    L.append(f"")
    
    return "\n".join(L)


def _emit_legacy_embedding(L, s):
    """Emit embedding layer using the legacy pattern."""
    L.append(f"  // Create embedding layer")
    if s.tie_word_embeddings:
        L.append(f'  const std::string embedding_type = TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";')
    else:
        L.append(f'  const std::string embedding_type = "embedding_layer";')
    
    L.append(f'  layers.push_back(createLayer(embedding_type, {{')
    L.append(f'    withKey("name", "embedding0"),')
    L.append(f'    withKey("in_dim", NUM_VOCAB),')
    L.append(f'    withKey("out_dim", DIM)')
    L.append(f"  }}));")
    L.append(f"")


def _emit_legacy_encoder_decoder_blocks(L, s, norm_type):
    """Emit encoder + decoder blocks using the legacy pattern."""
    # This would need to be implemented based on the specific structure
    # For now, just emit a comment
    L.append(f"  // TODO: Implement encoder-decoder blocks")
    L.append(f"  // This requires implementing createEncoderBlock and createDecoderBlock methods")
    L.append(f"")


def _emit_legacy_hybrid_blocks(L, s, block_type, blocks_info):
    """Emit hybrid blocks using the legacy pattern."""
    op_type_list = blocks_info["op_type_list"]
    L.append(f"  // Create transformer blocks")
    L.append(f"  for (int i = 0; i < NUM_LAYERS; ++i) {{")
    L.append(f"    std::vector<LayerHandle> transformer;")
    
    # Always generate standard block code for single operator type
    # Only show TODO for true hybrid models with multiple operator types
    if len(op_type_list) > 1:
        # Hybrid model with multiple operator types
        L.append(f"    // TODO: Implement hybrid block logic")
        L.append(f"    // This requires per-layer type dispatch")
        L.append(f"    // For now, using standard block as fallback")
    # Standard transformer blocks (for both hybrid fallback and non-hybrid models)
    L.append(f"    if (i == 0) {{")
    L.append(f"      transformer = createTransformer{block_type}(0, \"embedding0\");")
    L.append(f"    }} else {{")
    L.append(f"      transformer = createTransformer{block_type}(i, ")
    L.append(f"        \"layer\" + std::to_string(i - 1) + \"_decoder_output\");")
    L.append(f"    }}")
    L.append(f"    layers.insert(layers.end(), transformer.begin(), transformer.end());")
    
    L.append(f"  }}")
    L.append(f"")


def _emit_legacy_standard_blocks(L, s, block_type):
    """Emit standard transformer blocks using the legacy pattern."""
    L.append(f"  // Create transformer blocks")
    L.append(f"  for (int i = 0; i < NUM_LAYERS; ++i) {{")
    L.append(f"    std::vector<LayerHandle> transformer;")
    L.append(f"    if (i == 0) {{")
    L.append(f"      transformer = createTransformer{block_type}(0, \"embedding0\");")
    L.append(f"    }} else {{")
    L.append(f"      transformer = createTransformer{block_type}(i, ")
    L.append(f"        \"layer\" + std::to_string(i - 1) + \"_decoder_output\");")
    L.append(f"    }}")
    L.append(f"    layers.insert(layers.end(), transformer.begin(), transformer.end());")
    L.append(f"  }}")
    L.append(f"")


def _emit_legacy_final_norm(L, s, norm_type):
    """Emit final normalization layer using the legacy pattern."""
    L.append(f"  // Create final RMS norm")
    L.append(f'  layers.push_back(createLayer("{norm_type}", {{')
    L.append(f'    withKey("name", "output_norm"),')
    L.append(f'    withKey("epsilon", std::to_string(NORM_EPS)),')
    if norm_type == "rms_norm":
        L.append(f'    withKey("packed", "false"),')
    L.append(f'    withKey("input_layers", "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output")')
    L.append(f"  }}));")
    L.append(f"")


def _emit_legacy_lm_head(L, s):
    """Emit LM head layer using the legacy pattern."""
    L.append(f"  // Create LM head")
    if s.tie_word_embeddings:
        L.append(f'  const std::string lmhead_type = TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "fully_connected";')
    else:
        L.append(f'  const std::string lmhead_type = "fully_connected";')
    
    L.append(f'  layers.push_back(createLayer(lmhead_type, {{')
    L.append(f'    withKey("name", "lm_head"),')
    L.append(f'    withKey("unit", NUM_VOCAB),')
    L.append(f'    withKey("disable_bias", "true"),')
    L.append(f'    withKey("input_layers", "output_norm")')
    if s.tie_word_embeddings:
        L.append(f'    , withKey("shared_from", "embedding0")')
    L.append(f"  }}));")
    L.append(f"")