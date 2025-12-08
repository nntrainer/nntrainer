gpt_oss_src = [
  meson.current_source_dir() / 'gptoss_causallm.cpp',
  meson.current_source_dir() / 'gptoss_cached_slim_causallm.cpp',
]

gpt_oss_inc = include_directories('.')

# Define Layers
causallm_gptoss_moe_layer_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer.cpp']
causallm_gptoss_moe_layer_cached_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer_cached.cpp']

causallm_gptoss_moe_layer = shared_library(
    'gptoss_moe_layer',
    causallm_gptoss_moe_layer_src_abs,
    include_directories: [causallm_layer_inc, gpt_oss_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

casuallm_gptoss_moe_layer_dep = declare_dependency(
    link_with: causallm_gptoss_moe_layer,
    include_directories: gpt_oss_inc
)

causallm_cached_slim_gptoss_moe_layer = shared_library(
    'cached_slim_gptoss_moe_layer',
    causallm_gptoss_moe_layer_cached_src_abs,
    include_directories: [causallm_layer_inc, gpt_oss_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_cached_slim_gpt_oss_moe_layer_dep = declare_dependency(
    link_with: causallm_cached_slim_gptoss_moe_layer,
    include_directories: gpt_oss_inc
)

causallm_src += gpt_oss_src
causallm_inc += gpt_oss_inc
causallm_layer_dependencies += [casuallm_gptoss_moe_layer_dep, causallm_cached_slim_gpt_oss_moe_layer_dep]
