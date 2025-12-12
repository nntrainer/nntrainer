causallm_src = [
    meson.current_source_dir() / 'causal_lm.cpp',
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
]

causallm_src += [
    meson.current_source_dir() / 'qwen3_causallm.cpp',
    meson.current_source_dir() / 'qwen3_moe_causallm.cpp',
    meson.current_source_dir() / 'qwen3_slim_moe_causallm.cpp',
    meson.current_source_dir() / 'qwen3_cached_slim_moe_causallm.cpp',
    meson.current_source_dir() / 'nntr_qwen3_causallm.cpp',
    meson.current_source_dir() / 'nntr_qwen3_moe_causallm.cpp',
    meson.current_source_dir() / 'gptoss_causallm.cpp',
    meson.current_source_dir() / 'gptoss_cached_slim_causallm.cpp',
    meson.current_source_dir() / 'ernie_causallm.cpp',
]

executable_src = [
    meson.current_source_dir() / 'main.cpp',
]

causallm_inc_abs = [meson.current_source_dir()]
causallm_inc = [include_directories('.')]

# Build layer dependency
subdir('layers')

# causallm_layer_dependencies
causallm_layer_dependencies = [
    causallm_rms_norm_dep,
    causallm_tie_word_embedding_dep,
    causallm_swiglu_dep,
    causallm_mha_core_dep,
    causallm_embedding_layer_dep,
    causallm_reshaped_rms_norm_dep,
    causallm_moe_layer_dep,
    causallm_slim_moe_layer_dep,
    causallm_cached_slim_moe_layer_dep,
    causallm_qkv_layer_dep, 
    casuallm_gptoss_moe_layer_dep,
    causallm_cached_slim_gpt_oss_moe_layer_dep,
    causallm_ernie_moe_layer_dep,
]

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.source_root(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'

cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

causallm = shared_library('causallm',
    causallm_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        causallm_layer_dependencies, 
        tokenizer_dependencies,
        openmp_dep
    ],
    include_directories: causallm_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: cpp_args
)

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)
