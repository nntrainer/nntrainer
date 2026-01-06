# Source files for XLMRoberta application
executable_src = [
    meson.current_source_dir() / 'main.cpp',
]

# Include directories
xmlroberta_inc = [include_directories('.')]

# Use CausalLM shared library from Applications/CausalLM
# This avoids duplication and ensures consistency

e = executable('nntr_xmlroberta',
    executable_src,
    include_directories: xmlroberta_inc,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        # CausalLM dependencies - these are defined in Applications/CausalLM
        causallm_rms_norm_dep,
        causallm_tie_word_embedding_dep,
        causallm_swiglu_dep,
        causallm_mha_core_dep,
        causallm_embedding_layer_dep,
        causallm_reshaped_rms_norm_dep,
        causallm_qkv_layer_dep,
        # CausalLM shared library dependency
        causallm_dep
    ],
)
