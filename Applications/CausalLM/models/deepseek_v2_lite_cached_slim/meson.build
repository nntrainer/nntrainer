deepseek_v2_lite_slim_src = [
    meson.current_source_dir() / 'deepseek_v2_lite_slim_causallm.cpp',
]

deepseek_v2_lite_slim_inc = include_directories('.')

# Define Layers
causallm_moe_layer_src_abs = [meson.current_source_dir() / 'deepseek_moe_layer_slim.cpp']

causallm_moe_layer = shared_library(
    'deepseek_moe_layer_slim',
    causallm_moe_layer_src_abs,
    include_directories: [causallm_layer_inc, deepseek_v2_lite_slim_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_moe_layer_dep = declare_dependency(
    link_with: causallm_moe_layer,
    include_directories: deepseek_v2_lite_slim_inc
)

causallm_src += deepseek_v2_lite_slim_src
causallm_inc += deepseek_v2_lite_slim_inc
causallm_layer_dependencies += [causallm_moe_layer_dep]
