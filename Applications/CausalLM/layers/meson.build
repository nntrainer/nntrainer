# '..' brings in the app-root headers the layers share with the models --
# today kv_ring.h, the one definition of the KV-ring capacity rule that
# mha_core.cpp and models/transformer.h must not disagree about.
causallm_layer_inc_abs = [meson.current_source_dir(), meson.current_source_dir() / '..']
causallm_layer_inc = [include_directories('.'), include_directories('..')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_rms_norm_gpu_src_abs = [meson.current_source_dir() / 'rms_norm_gpu.cpp']
causallm_causal_conv1d_src_abs = [meson.current_source_dir() / 'causal_conv1d_layer.cpp']
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
causallm_rms_reverse_norm_src_abs = [meson.current_source_dir() / 'rms_reverse_norm.cpp']
causallm_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
causallm_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
causallm_deberta_attention_src_abs = [meson.current_source_dir() / 'deberta_attention_layer.cpp']
causallm_shared_fully_connected_src_abs = [meson.current_source_dir() / 'shared_fully_connected_layer.cpp']
causallm_per_layer_slice_src_abs = [meson.current_source_dir() / 'per_layer_slice.cpp', meson.current_source_dir() / 'per_layer_slice_gpu.cpp']
causallm_custom_multiply_layer_src_abs = [meson.current_source_dir() / 'custom_multiply.cpp']

causallm_rms_norm = shared_library(
    'rms_norm_layer', 
    causallm_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_rms_norm_dep = declare_dependency(
    link_with: causallm_rms_norm,
    include_directories: causallm_layer_inc
)

# GPU-routed RMSNorm (RMSNormLayerGPU). Referenced unconditionally by
# transformer.cpp; the GPU dispatch inside it only fires under enable-opencl.
causallm_rms_norm_gpu = shared_library(
    'rms_norm_gpu_layer',
    causallm_rms_norm_gpu_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_rms_norm_gpu_dep = declare_dependency(
    link_with: causallm_rms_norm_gpu,
    include_directories: causallm_layer_inc
)


causallm_causal_conv1d = shared_library(
    'causal_conv1d_layer',
    causallm_causal_conv1d_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_causal_conv1d_dep = declare_dependency(
    link_with: causallm_causal_conv1d,
    include_directories: causallm_layer_inc
)



causallm_mha_core = shared_library(
    'mha_core_layer',
    causallm_mha_core_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_mha_core_dep = declare_dependency(
    link_with: causallm_mha_core,
    include_directories: causallm_layer_inc
)

causallm_embedding_layer = shared_library(
    'embedding_layer',
    causallm_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_layer_dep = declare_dependency(
    link_with: causallm_embedding_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_pooling_layer = shared_library(
    'embedding_pooling_layer',
    causallm_embedding_pooling_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_pooling_layer_dep = declare_dependency(
    link_with: causallm_embedding_pooling_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_normalize_layer = shared_library(
    'embedding_normalize_layer',
    causallm_embedding_normalize_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_embedding_normalize_layer_dep = declare_dependency(
    link_with: causallm_embedding_normalize_layer,
    include_directories: causallm_layer_inc
)

causallm_reshaped_rms_norm = shared_library(
    'reshaped_rms_norm_layer', 
    causallm_reshaped_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_reshaped_rms_norm_dep = declare_dependency(
    link_with: causallm_reshaped_rms_norm,
    include_directories: causallm_layer_inc
)


causallm_deberta_attention_layer = shared_library(
    'deberta_attention_layer',
    causallm_deberta_attention_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_deberta_attention_layer_dep = declare_dependency(
    link_with: causallm_deberta_attention_layer,
    include_directories: causallm_layer_inc
)

causallm_shared_fully_connected_layer = shared_library(
    'shared_fully_connected_layer',
    causallm_shared_fully_connected_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_shared_fully_connected_layer_dep = declare_dependency(
    link_with: causallm_shared_fully_connected_layer,
    include_directories: causallm_layer_inc
)

causallm_rms_reverse_norm = shared_library(
    'rms_reverse_norm_layer',
    causallm_rms_reverse_norm_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_rms_reverse_norm_dep = declare_dependency(
    link_with: causallm_rms_reverse_norm,
    include_directories: causallm_layer_inc
)

causallm_per_layer_slice = shared_library(
    'per_layer_slice_layer',
    causallm_per_layer_slice_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_per_layer_slice_dep = declare_dependency(
    link_with: causallm_per_layer_slice,
    include_directories: causallm_layer_inc
)



causallm_custom_multiply_layer = shared_library(
    'custom_multiply',
    causallm_custom_multiply_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_custom_multiply_dep = declare_dependency(
    link_with: causallm_custom_multiply_layer,
    include_directories: causallm_layer_inc
)
