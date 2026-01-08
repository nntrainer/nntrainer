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
        # CausalLM shared library dependency
        causallm_dep
    ],
)
