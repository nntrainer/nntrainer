gemma4_src = [
    meson.current_source_dir() / 'gemma4_causallm.cpp',
    meson.current_source_dir() / 'gemma4_moe_causallm.cpp',
    meson.current_source_dir() / 'gemma4_htp_causallm.cpp',
    meson.current_source_dir() / 'gemma4_htp_moe_causallm.cpp',
    meson.current_source_dir() / 'gemma4_htp_weights.cpp',
]

gemma4_inc = include_directories('.')

gemma4_moe_layer_src_abs = [meson.current_source_dir() / 'gemma4_moe_layer.cpp']
# The gpu-engine variant of the layer (experts on the OpenCL device).
if get_option('enable-opencl')
    gemma4_moe_layer_src_abs += meson.current_source_dir() / 'gemma4_moe_layer_gpu.cpp'
endif

gemma4_moe_layer = shared_library(
    'gemma4_moe_layer',
    gemma4_moe_layer_src_abs,
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

causallm_inc += include_directories('../lfm2_moe')
causallm_src += files('../../layers/htp_mha_core.cpp',
                      '../../layers/htp_qkv_layer.cpp',
                      '../lfm2_moe/lfm2_moe_layer.cpp',
                      '../lfm2_moe/lfm2_htp_moe_causallm.cpp',
                      '../lfm2_moe/lfm2_moe_pool_layer.cpp')
causallm_layer_dependencies += [causallm_residual_add_dep,
  causallm_dense_ffn_layer_dep, causallm_conv_block_layer_dep]
