causallm_layer_inc_abs = [meson.current_source_dir()]
causallm_layer_inc = [include_directories('.')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_swiglu_src_abs = [meson.current_source_dir() / 'swiglu.cpp']
causallm_tie_word_embedding_src_abs = [meson.current_source_dir() / 'tie_word_embedding.cpp']
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
causallm_moe_layer_src_abs = [meson.current_source_dir() / 'qwen_moe_layer.cpp']
causallm_slim_moe_layer_src_abs = [meson.current_source_dir() / 'qwen_moe_layer_fsu.cpp']
causallm_cached_slim_moe_layer_src_abs = [meson.current_source_dir() / 'qwen_moe_layer_cached.cpp']
causallm_qkv_layer_src_abs = [meson.current_source_dir() / 'qkv_layer.cpp']
causallm_gptoss_moe_layer_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer.cpp']
causallm_gptoss_moe_layer_cached_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer_cached.cpp']
causallm_ernie_moe_layer_src_abs = [meson.current_source_dir() / 'ernie_moe_layer.cpp']

openmp_dep = dependency('openmp')

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

causallm_swiglu = shared_library(
    'swiglu_layer',
    causallm_swiglu_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_swiglu_dep = declare_dependency(
    link_with: causallm_swiglu,
    include_directories: causallm_layer_inc
)

causallm_tie_word_embedding = shared_library(
    'tie_word_embedding_layer',
    causallm_tie_word_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_tie_word_embedding_dep = declare_dependency(
    link_with: causallm_tie_word_embedding,
    include_directories: causallm_layer_inc
)

causallm_mha_core = shared_library(
    'mha_core_layer',
    causallm_mha_core_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
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
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_layer_dep = declare_dependency(
    link_with: causallm_embedding_layer,
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

causallm_moe_layer = shared_library(
    'qwen_moe_layer',
    causallm_moe_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_moe_layer_dep = declare_dependency(
    link_with: causallm_moe_layer,
    include_directories: causallm_layer_inc
)

causallm_slim_moe_layer = shared_library(
    'qwen_moe_layer_fsu',
    causallm_slim_moe_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_slim_moe_layer_dep = declare_dependency(
    link_with: causallm_slim_moe_layer,
    include_directories: causallm_layer_inc
)

causallm_cached_slim_moe_layer = shared_library(
    'qwen_moe_layer_cached',
    causallm_cached_slim_moe_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_cached_slim_moe_layer_dep = declare_dependency(
    link_with: causallm_cached_slim_moe_layer,
    include_directories: causallm_layer_inc
)

causallm_qkv_layer = shared_library(
    'qkv_layer',
    causallm_qkv_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_qkv_layer_dep = declare_dependency(
    link_with: causallm_qkv_layer,
    include_directories: causallm_layer_inc
)

causallm_gptoss_moe_layer = shared_library(
    'gptoss_moe_layer',
    causallm_gptoss_moe_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

casuallm_gptoss_moe_layer_dep = declare_dependency(
    link_with: causallm_gptoss_moe_layer,
    include_directories: causallm_layer_inc
)

causallm_cached_slim_gptoss_moe_layer = shared_library(
    'cached_slim_gptoss_moe_layer',
    causallm_gptoss_moe_layer_cached_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_cached_slim_gpt_oss_moe_layer_dep = declare_dependency(
    link_with: causallm_cached_slim_gptoss_moe_layer,
    include_directories: causallm_layer_inc
)

causallm_ernie_moe_layer = shared_library(
    'ernie_moe_layer',
    causallm_ernie_moe_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_ernie_moe_layer_dep = declare_dependency(
    link_with: causallm_ernie_moe_layer,
    include_directories: causallm_layer_inc
)
