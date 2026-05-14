causallm_src = [
    meson.current_source_dir() / 'chat_template.cpp',
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
    meson.current_source_dir() / 'api' / 'causal_lm_api.cpp',
    meson.current_source_dir() / 'api' / 'model_config.cpp',
    meson.current_source_dir() / 'kv_cache_manager.cpp',
]

executable_src = [
    meson.current_source_dir() / 'main.cpp',
]

causallm_inc_abs = [meson.current_source_dir()]
causallm_inc = [include_directories('.')]

# Build layer dependency
causallm_layer_dependencies = []
subdir('layers')

# Add common layers to dependencies
causallm_layer_dependencies += [
    causallm_rms_norm_dep,
    causallm_tie_word_embedding_dep,
    causallm_lm_head_dep,
    causallm_swiglu_dep,
    causallm_mha_core_dep,
    causallm_embedding_layer_dep,
    causallm_lm_head_dep,
    causallm_reshaped_rms_norm_dep,
    causallm_qkv_layer_dep,
    causallm_embedding_pooling_layer_dep,
    causallm_embedding_normalize_layer_dep,
]

subdir('models')

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.source_root(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'
tokenizer_windows_path = meson.build_root() / 'tokenizers_c_win' / 'target' / 'release' / 'tokenizers_c.lib'
tokenizer_windows_link_args = [
    tokenizer_windows_path,
    'bcrypt.lib',
    'ntdll.lib',
    'userenv.lib',
    'ws2_32.lib'
]

cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

if get_option('platform') == 'windows'
    causallm = static_library('causallm',
        causallm_src,
        dependencies: [
            nntrainer_dep,
            nntrainer_ccapi_dep,
            causallm_layer_dependencies,
            tokenizer_dependencies
        ],
        include_directories: causallm_inc,
        install: true,
        cpp_args: cpp_args
    )
else
    causallm = shared_library('causallm',
        causallm_src,
        dependencies: [
            nntrainer_dep,
            nntrainer_ccapi_dep,
            causallm_layer_dependencies,
            tokenizer_dependencies
        ],
        include_directories: causallm_inc,
        install: true,
        install_dir: application_install_dir,
        link_args: [tokenizer_path],
        cpp_args: cpp_args
    )
endif

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc,
    link_args: get_option('platform') == 'windows' ? tokenizer_windows_link_args : []
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)


quantize_src = [
    meson.current_source_dir() / 'quantize.cpp',
]

e_quantize = executable('nntr_quantize',
    quantize_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)


test_api_src = [
    meson.current_source_dir() / 'api' / 'test_api.cpp',
]

test_api = executable('test_api',
    test_api_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
    build_by_default: true,
    install: false,
)

# KVCacheManager unit tests
if get_option('enable-test')
  unittest_kv_cache_manager = executable(
    'unittest_kv_cache_manager',
    meson.source_root() / 'test' / 'unittest' / 'layers' / 'unittest_kv_cache_manager.cpp',
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, gtest_dep, causallm_dep],
    include_directories: causallm_inc,
    build_by_default: true,
    install: false,
  )

  test('unittest_kv_cache_manager', unittest_kv_cache_manager,
    args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_kv_cache_manager'),
    timeout: 60,
    suite: 'unittests'
  )
endif
