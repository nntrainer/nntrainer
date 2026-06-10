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
causallm_inc_minja_inc = include_directories('third_party/minja/include')
causallm_inc_nlohmann_inc = include_directories('third_party')

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
    causallm_deberta_attention_layer_dep,
    causallm_shared_fully_connected_layer_dep,
    causallm_per_layer_slice_dep,
    causallm_scalar_multiply_dep,
    causallm_logit_softcapping_dep,
]

subdir('models')

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.source_root(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_dependencies = []
tokenizer_option = get_option('causallm-tokenizer-lib')
tokenizer_rust_target = ''
if get_option('platform') == 'windows'
    if meson.is_cross_build()
        if host_machine.cpu_family() == 'aarch64'
            tokenizer_rust_target = 'aarch64-pc-windows-msvc'
        elif host_machine.cpu_family() == 'x86_64'
            tokenizer_rust_target = 'x86_64-pc-windows-msvc'
        endif
    endif

    if tokenizer_option != ''
        tokenizer_path = tokenizer_option
    elif tokenizer_rust_target == ''
        tokenizer_path = meson.build_root() / 'tokenizers_c_win' / 'target' / 'release' / 'tokenizers_c.lib'
    else
        tokenizer_path = meson.build_root() / 'tokenizers_c_win' / 'target' / tokenizer_rust_target / 'release' / 'tokenizers_c.lib'
    endif
else
    if tokenizer_option != ''
        error('causallm-tokenizer-lib is only supported for Windows CausalLM builds.')
    endif

    tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'
endif

generate_windows_tokenizer = tokenizer_option == ''
generate_windows_tokenizer = generate_windows_tokenizer and get_option('platform') == 'windows'
generate_windows_tokenizer = generate_windows_tokenizer and build_machine.system() == 'windows'
generate_windows_tokenizer = generate_windows_tokenizer and not fs.exists(tokenizer_path)

if generate_windows_tokenizer
    run_command(
        powershell_prog,
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        meson.current_source_dir() / 'build_tokenizer_windows.ps1',
        '-BuildDir',
        meson.build_root(),
        '-RustTarget',
        tokenizer_rust_target,
        check: true
    )
endif

if not fs.exists(tokenizer_path)
    if get_option('platform') == 'windows'
        tokenizer_error = 'CausalLM Windows build requires tokenizers_c.lib. Run Applications\\CausalLM\\build_tokenizer_windows.ps1 -BuildDir @0@ or pass -Dcausallm-tokenizer-lib=/path/to/tokenizers_c.lib.'
        error(tokenizer_error.format(meson.build_root()))
    else
        error('CausalLM tokenizer library not found: @0@'.format(tokenizer_path))
    endif
endif

tokenizer_link_args = [tokenizer_path]
if get_option('platform') == 'windows'
    tokenizer_link_args += [
        'bcrypt.lib',
        'ntdll.lib',
        'userenv.lib',
        'ws2_32.lib'
    ]
endif

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
        include_directories: [
            causallm_inc,
            causallm_inc_minja_inc,
            causallm_inc_nlohmann_inc
        ],
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
        include_directories: [
            causallm_inc,
            causallm_inc_minja_inc,
            causallm_inc_nlohmann_inc
        ],
        install: true,
        install_dir: application_install_dir,
        link_args: tokenizer_link_args,
        cpp_args: cpp_args
    )
endif

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc,
    link_args: get_option('platform') == 'windows' ? tokenizer_link_args : []
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: [causallm_inc, causallm_inc_minja_inc, causallm_inc_nlohmann_inc],
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
  causallm_unittest_inc = include_directories(
    '../../test/unittest/models'
  )

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

  # Ubuntu native builds use platform=none.
  model_unittest_platforms = ['none', 'windows', 'android']
  platform = get_option('platform')
  enable_causallm_model_unittest = platform in model_unittest_platforms
  if enable_causallm_model_unittest
    unittest_causallm_models = executable(
      'unittest_causallm_models',
      [
        meson.source_root() / 'test' / 'unittest' / 'models' / 'causallm_test_utils.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gemma3.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen2.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3.cpp',
      ],
      dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        gtest_main_dep,
        causallm_layer_dependencies,
        causallm_dep
      ],
      include_directories: causallm_inc + [causallm_unittest_inc],
      build_by_default: true,
      install: false,
    )

    test('unittest_causallm_models', unittest_causallm_models,
      args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_causallm_models'),
      timeout: 60,
      suite: 'unittests'
    )
  endif
endif
