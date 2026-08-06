# api/{causal_lm_api,model_config,streamer,callback_streamer}.cpp were removed
# here: the API layer is now quick_dot_ai_api, a separate lib there. NOTE:
# models/causal_lm.cpp still calls streamer_put()/streamer_end() (declared in
# api/streamer.h). streamer.cpp is self-contained (only depends on
# streamer.h, no other api/ sources), so it is re-added directly to
# causallm_src below.
#
# nntrainer standalone app build: the public extension that used to be
# QuickAI-only (xgrammar + the QNN model-runtime infra + the public
# gemma4-e2b-qnn model) is folded into libcausallm.so here, so this
# submodule's own meson build can produce the full app — CPU always, QNN
# opt-in via enable-qnn-models — without QuickAI's root meson project:
#   - xgrammar (api/xgrammar_wrapper.cpp, api/xgrammar_manager.cpp + the
#     vendored xgrammar/cpp/*.cc library) compiles into libcausallm.so
#     exactly once (see the xgrammar block below causallm_src) and is
#     link_whole'd in, so the XGrammarManager Meyers singleton
#     (api/xgrammar_manager.cpp:26) is defined+exported a single time.
#     CPU-only builds get it too — it isn't QNN-specific.
#   - The QNN model-runtime infra (qnn/*.cpp) + the public gemma4-e2b-qnn
#     model compile into libcausallm.so when enable-qnn-models is set (see
#     the QNN block below causallm_src). Proprietary QNN models are layered
#     on top by a separate downstream project, not in this public tree.
#
# The enable-api block further below provides the libquick_dot_ai_api.so
# target (+ quick_dot_ai_test) mirroring QuickAI's api/meson.build. It no
# longer compiles xgrammar itself — it borrows the single copy from
# libcausallm.so via causallm_dep. ENABLE_QNN_MODELS is defined for it only
# when enable-qnn-models is set, matching the QNN sources already compiled
# into libcausallm.so above.
causallm_src = [
    meson.current_source_dir() / 'chat_template.cpp',
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
    meson.current_source_dir() / 'kv_cache_manager.cpp',
    meson.current_source_dir() / 'api' / 'streamer.cpp',
    # factory.cpp: the ONE out-of-line Factory::Instance() definition. Must be
    # compiled into libcausallm exactly once so every consumer .so (the api lib
    # and the optional model plugin) shares a single Factory (see factory.h).
    meson.current_source_dir() / 'factory.cpp',
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
    causallm_causal_conv1d_dep,
    causallm_custom_multiply_dep,
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

# ── xgrammar (wrapper + vendored library), compiled into libcausallm.so ──
# api/xgrammar_manager.cpp:26 defines a Meyers singleton
# (XGrammarManager::Instance()); compiling it into more than one shared
# library risks two live instances. Build it once here as a static library
# and link_whole it into libcausallm.so below, so there's a single
# definition that the api layer (E4, further down) and any downstream model
# plugin borrow at runtime via causallm_dep. CPU-only builds get xgrammar
# too — it isn't QNN-specific.
causallm_xgrammar_inc = [
    causallm_inc,
    causallm_inc_minja_inc,
    causallm_inc_nlohmann_inc,
    include_directories('api'),
    include_directories('xgrammar' / 'include'),
    include_directories('xgrammar' / '3rdparty' / 'picojson'),
    include_directories('xgrammar' / '3rdparty' / 'dlpack' / 'include'),
]

causallm_xgrammar_src = [
    # wrapper: bridges causallm::XGrammar to the xgrammar library.
    meson.current_source_dir() / 'api' / 'xgrammar_wrapper.cpp',
    meson.current_source_dir() / 'api' / 'xgrammar_manager.cpp',
    # xgrammar library sources (from the xgrammar submodule checkout under
    # this directory).
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'compiled_grammar.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'config.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'earley_parser.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'fsm_builder.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'fsm.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_builder.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_compiler.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_functor.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_matcher.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_parser.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar_printer.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'grammar.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'json_schema_converter_ext.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'json_schema_converter.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'regex_converter.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'structural_tag.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'tokenizer_info.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'testing.cc',
    # Support files (required for linking)
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'support' / 'logging.cc',
    meson.current_source_dir() / 'xgrammar' / 'cpp' / 'support' / 'recursion_guard.cc',
]

causallm_xgrammar_lib = static_library('causallm_xgrammar',
    causallm_xgrammar_src,
    include_directories: causallm_xgrammar_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    # The vendored xgrammar sources were only ever compiled under QuickAI's
    # relaxed werror=false project setting and are not warning-clean under
    # nntrainer's stricter werror=true. Relax werror for this static library
    # only; libcausallm's own sources keep werror=true.
    override_options: ['werror=false'],
)

# Extra include dirs the causallm target needs on top of causallm_inc
# (xgrammar always; QNN only when enable-qnn-models, see below).
causallm_extra_inc = [causallm_xgrammar_inc]

# ── QNN model-runtime infra + public QNN model (gated) ───────────────────
# Requires enable-npu: the QNN backend (libqnn_context.so) is registered
# into nntrainer's Engine at runtime by qnn/quick_dot_ai_qnn.cpp
# (Engine::Global().registerContext("libqnn_context.so", ...)), not linked
# directly, so no QNN SDK headers are needed to compile these sources.
if get_option('enable-qnn-models')
    if not get_option('enable-npu')
        error('enable-qnn-models requires enable-npu (the libqnn_context.so QNN backend)')
    endif

    causallm_src += [
        meson.current_source_dir() / 'qnn' / 'android_memory_allocator.cpp',
        meson.current_source_dir() / 'qnn' / 'quick_dot_ai_qnn.cpp',
        meson.current_source_dir() / 'qnn' / 'graph_parser.cpp',
        meson.current_source_dir() / 'qnn' / 'generate_qnn_utils.cpp',
        meson.current_source_dir() / 'qnn' / 'qnn_kv_cache_manager.cpp',
        meson.current_source_dir() / 'qnn' / 'gemma4-e2b-qnn' / 'gemma4_e2b_qnn.cpp',
    ]

    causallm_extra_inc += [
        include_directories('qnn'),
        include_directories('qnn' / 'gemma4-e2b-qnn'),
    ]

    cpp_args += [
        '-DENABLE_QNN=1',
        '-DENABLE_QNN_MODELS=1',
    ]
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
            causallm_inc_nlohmann_inc,
            causallm_extra_inc
        ],
        link_whole: [causallm_xgrammar_lib],
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
            causallm_inc_nlohmann_inc,
            causallm_extra_inc
        ],
        install: true,
        install_dir: application_install_dir,
        link_args: tokenizer_link_args,
        link_whole: [causallm_xgrammar_lib],
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


safetensors_info_src = [
    meson.current_source_dir() / 'safetensors_info.cpp',
]

e_safetensors_info = executable('nntr_safetensors_info',
    safetensors_info_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep],
)


# ── E4: libquick_dot_ai_api.so (+ quick_dot_ai_test smoke exe) ──────────
# Mirrors QuickAI's api/meson.build: xgrammar is compiled once into
# libcausallm.so (see the xgrammar block above causallm_src) and this target
# borrows the single XGrammarManager singleton from it via causallm_dep,
# instead of compiling its own copy. ENABLE_QNN_MODELS is defined only when
# enable-qnn-models is set, matching the QNN sources conditionally compiled
# into libcausallm.so above — the dynamic_cast<Quick_Dot_AI_QNN *> /
# make_unique<Gemma4_E2B_QNN> paths those defines unlock in
# api/quick_dot_ai_api.cpp then resolve at link via causallm_dep.
if get_option('enable-api')
  quick_dot_ai_api_inc = [
    causallm_inc,
    causallm_inc_minja_inc,
    causallm_inc_nlohmann_inc,
    include_directories('api'),
    include_directories('xgrammar' / 'include'),
    include_directories('xgrammar' / '3rdparty' / 'picojson'),
    include_directories('xgrammar' / '3rdparty' / 'dlpack' / 'include'),
  ]

  quick_dot_ai_api_cpp_args = []
  if get_option('enable-qnn-models')
    quick_dot_ai_api_inc += [
      include_directories('qnn'),
      include_directories('qnn' / 'gemma4-e2b-qnn'),
    ]
    quick_dot_ai_api_cpp_args += ['-DENABLE_QNN_MODELS=1']
  endif

  quick_dot_ai_api_src = [
    meson.current_source_dir() / 'api' / 'quick_dot_ai_api.cpp',
    meson.current_source_dir() / 'api' / 'model_callbacks.cpp',
    meson.current_source_dir() / 'api' / 'model_config.cpp',
    meson.current_source_dir() / 'api' / 'model_descriptors_public.cpp',
    meson.current_source_dir() / 'api' / 'streamer.cpp',
    meson.current_source_dir() / 'api' / 'callback_streamer.cpp',
  ]

  quick_dot_ai_api_lib = shared_library('quick_dot_ai_api',
    quick_dot_ai_api_src,
    include_directories: quick_dot_ai_api_inc,
    dependencies: [nntrainer_dep, causallm_dep, dependency('threads')],
    cpp_args: quick_dot_ai_api_cpp_args,
    install: true,
  )

  quick_dot_ai_api_dep = declare_dependency(
    link_with: quick_dot_ai_api_lib,
    include_directories: include_directories('api'),
    dependencies: [causallm_dep],
  )

  executable('quick_dot_ai_test',
    meson.current_source_dir() / 'api-app' / 'test_api.cpp',
    dependencies: [quick_dot_ai_api_dep],
    install: false,
  )
endif

# KVCacheManager unit tests
if get_option('enable-test')
  causallm_unittest_inc = include_directories(
    '../../test/unittest/models'
  )

  unittest_embedding_sidecar_lut = executable(
    'unittest_embedding_sidecar_lut',
    meson.current_source_dir() / 'layers' / 'unittest_embedding_sidecar_lut.cpp',
    dependencies: [nntrainer_dep, gtest_main_dep, causallm_embedding_layer_dep],
    include_directories: causallm_layer_inc,
    build_by_default: true,
    install: false,
  )

  test('unittest_embedding_sidecar_lut', unittest_embedding_sidecar_lut,
    args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_embedding_sidecar_lut'),
    timeout: 60,
    suite: 'unittests'
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

  # Extract causallm_reference fixture binaries from packaging/
  causallm_fixture_tars = [
    meson.source_root() / 'packaging' / 'causallm_reference_generation.tar.gz',
    meson.source_root() / 'packaging' / 'causallm_reference_embedding.tar.gz',
  ]
  # The tarballs already contain a top-level `causallm_reference/` directory, so
  # extract into models/ (NOT models/causallm_reference/, which would double the
  # path to models/causallm_reference/causallm_reference/ and leave the fixtures
  # where the tests cannot find them — causing every differential test to skip).
  causallm_fixture_dest = meson.source_root() / 'test' / 'unittest' / 'models'
  foreach fixture_tar : causallm_fixture_tars
    if build_machine.system() == 'windows'
      run_command([prog_win_cmd, '/C', 'tar', 'xzf',
                   fixture_tar.replace('/', '\\'),
                   '-C', causallm_fixture_dest.replace('/', '\\')], check: false)
    else
      run_command(['tar', 'xzf', fixture_tar, '-C', causallm_fixture_dest], check: false)
    endif
  endforeach

  unittest_callback_streamer = executable(
    'unittest_callback_streamer',
    [
      meson.current_source_dir() / 'api' / 'unittest_callback_streamer.cpp',
      meson.current_source_dir() / 'api' / 'callback_streamer.cpp',
      meson.current_source_dir() / 'api' / 'streamer.cpp',
    ],
    dependencies: [gtest_main_dep],
    include_directories: causallm_inc,
    build_by_default: true,
    install: false,
  )

  test('unittest_callback_streamer', unittest_callback_streamer,
    args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_callback_streamer'),
    timeout: 60,
    suite: 'unittests'
  )

  # NOTE: the 'unittest_cancel_api' target (api/unittest_cancel_api.cpp) was
  # removed here (UNVERIFIED — see note above test_api). It exercised the
  # removed causal_lm_api.{cpp,h} global cancel + '*ForTest' seams, which no
  # longer exist.

  # Ubuntu native builds use platform=none.
  model_unittest_platforms = ['none', 'windows', 'android']
  platform = get_option('platform')
  # Tiny model-level tests cover FP32 and Q4_0-FP32 correctness. Keep them out
  # of FP16 builds, where cache tensors intentionally use FP16 storage.
  enable_causallm_model_unittest = (
    platform in model_unittest_platforms and not get_option('enable-fp16')
  )
  if enable_causallm_model_unittest
    unittest_causallm_models = executable(
      'unittest_causallm_models',
      [
        meson.source_root() / 'test' / 'unittest' / 'models' / 'causallm_test_utils.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gemma3.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gemma3_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gemma4.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gemma4_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_moe.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_moe_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_slim_moe.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_cached_slim_moe.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gpt_oss.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_gpt_oss_cached_slim.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen2.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen2_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen3_embedding_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_qwen2_embedding_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_embedding_gemma_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_tinybert_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_deberta_v2_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_xlm_roberta_reference.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_lfm2.cpp',
        meson.source_root() / 'test' / 'unittest' / 'models' / 'unittest_causallm_lfm2_reference.cpp',
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
      suite: 'unittests',
      env: {'NNTR_QUANTIZE_BIN': e_quantize.full_path()}
    )
  endif
endif
