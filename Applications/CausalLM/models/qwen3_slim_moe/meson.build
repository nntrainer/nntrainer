qwen3_slim_moe_src = [
    meson.current_source_dir() / 'qwen3_slim_moe_causallm.cpp',
]

qwen3_slim_moe_inc = include_directories('.')

causallm_slim_moe_layer_src_abs = [meson.current_source_dir() / 'qwen_moe_layer_fsu.cpp']

causallm_slim_moe_layer = shared_library(
    'qwen_moe_layer_fsu',
    causallm_slim_moe_layer_src_abs,
    include_directories: [causallm_layer_inc, qwen3_slim_moe_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_slim_moe_layer_dep = declare_dependency(
    link_with: causallm_slim_moe_layer,
    include_directories: qwen3_slim_moe_inc
)

causallm_src += qwen3_slim_moe_src
causallm_inc += qwen3_slim_moe_inc
causallm_layer_dependencies += [causallm_slim_moe_layer_dep]
