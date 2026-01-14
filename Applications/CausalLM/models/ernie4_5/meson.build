ernie_src = [
    meson.current_source_dir() / 'ernie_causallm.cpp',
]

ernie_inc = include_directories('.')

# Define Layers
causallm_moe_layer_src_abs = [meson.current_source_dir() / 'ernie_moe_layer.cpp']

causallm_moe_layer = shared_library(
    'ernie_moe_layer',
    causallm_moe_layer_src_abs,
    include_directories: [causallm_layer_inc, ernie_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_moe_layer_dep = declare_dependency(
    link_with: causallm_moe_layer,
    include_directories: ernie_inc
)

causallm_src += ernie_src
causallm_inc += ernie_inc
causallm_layer_dependencies += [causallm_moe_layer_dep]
