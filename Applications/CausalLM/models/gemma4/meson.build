gemma4_src = [
    meson.current_source_dir() / 'gemma4_causallm.cpp',
    meson.current_source_dir() / 'gemma4_moe_causallm.cpp',
]

gemma4_inc = include_directories('.')

gemma4_moe_layer = shared_library(
    'gemma4_moe_layer',
    meson.current_source_dir() / 'gemma4_moe_layer.cpp',
    include_directories: [causallm_layer_inc, gemma4_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

gemma4_moe_layer_dep = declare_dependency(
    link_with: gemma4_moe_layer,
    include_directories: gemma4_inc
)

causallm_src += gemma4_src
causallm_inc += gemma4_inc
causallm_layer_dependencies += [gemma4_moe_layer_dep]
