# nntrainer app build: a self-contained meson project that
# builds the CausalLM app (libcausallm.so + libquick_dot_ai_api.so +
# quick_dot_ai_test) by linking a PREBUILT nntrainer engine via
# cxx.find_library() - exactly how QuickAI's root meson project works today,
# relocated into nntrainer and made Gauss-free.
#
# This project intentionally mirrors QuickAI's root meson.build / src/meson.build
# / api/meson.build / api-app/meson.build (proven, working techniques - sources
# outside the project root are referenced via STRING paths + '-I' compile args,
# NOT include_directories('..'), to avoid meson sandbox violations) while
# folding the public extension (xgrammar + QNN model-runtime infra + the public
# gemma4-e2b-qnn model) directly into libcausallm.so, the way
# Applications/CausalLM/meson.build does for the in-tree nntrainer build.
#
# Gauss-free: this project must never reference gauss models/descriptors. The
# proprietary gauss overlay is built by QuickAI's own meson project against
# the prebuilt libcausallm.so/libqnn_context.so produced here.
project('causallm-app', 'c', 'cpp',
  version: '0.2.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=false',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

# ── Compiler setup ──────────────────────────────────────────────────────
cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
fs = import('fs')

# ── Platform detection ──────────────────────────────────────────────────
_platform_opt = get_option('platform')
if _platform_opt == 'auto'
  is_android = host_machine.system() == 'android'
elif _platform_opt == 'android'
  is_android = true
else
  is_android = false
endif

# ── QNN guard: android only ─────────────────────────────────────────────
enable_qnn = get_option('enable-qnn')
if enable_qnn and not is_android
  warning('enable-qnn is only supported on android. Disabling QNN.')
  enable_qnn = false
endif

# ── Path configuration ───────────────────────────────────────────────────
# app_build/ lives at Applications/CausalLM/app_build; sources are one
# level up (Applications/CausalLM), the nntrainer submodule root is three
# levels up (app_build -> CausalLM -> Applications -> nntrainer).
app_build_root = meson.current_source_dir()
causallm_root   = app_build_root / '..'
nntrainer_root  = app_build_root / '..' / '..' / '..'

# ── Find pre-built nntrainer libraries ──────────────────────────────────
if is_android
  _nntr_result = nntrainer_root / 'builddir' / 'android_build_result'
  _nntr_libdir = _nntr_result / 'lib' / 'arm64-v8a'
  _nntr_incdir = _nntr_result / 'include' / 'nntrainer'

  nntrainer_abi_schema = meson.get_external_property(
    'nntrainer_abi_schema', 0)
  if nntrainer_abi_schema != 1
    error('nntrainer ABI metadata is missing or unsupported. Rebuild the ' +
          'Android engine and pass android_build_result/nntrainer-abi.ini ' +
          'as a cross file.')
  endif
  nntrainer_abi_cflags = meson.get_external_property(
    'nntrainer_abi_cflags', [])

  libnntrainer = cxx.find_library('nntrainer', dirs: [_nntr_libdir])
  libccapi     = cxx.find_library('ccapi-nntrainer', dirs: [_nntr_libdir])

  nntrainer_inc_args = ['-I' + _nntr_incdir]
else
  _nntr_builddir = nntrainer_root / get_option('nntrainer_builddir')

  libnntrainer = cxx.find_library('nntrainer',
    dirs: [_nntr_builddir / 'nntrainer'])
  libccapi     = cxx.find_library('ccapi-nntrainer',
    dirs: [_nntr_builddir / 'api' / 'ccapi'])

  # nntrainer source-tree includes (x86 build).
  # NOTE: Do NOT include the builddir itself - ruy/time.h shadows system <time.h>
  nntrainer_inc_args = []
  foreach p : [
    nntrainer_root / 'api' / 'ccapi' / 'include',
    nntrainer_root / 'api',
    nntrainer_root / 'nntrainer',
    nntrainer_root / 'nntrainer' / 'layers',
    nntrainer_root / 'nntrainer' / 'layers' / 'loss',
    nntrainer_root / 'nntrainer' / 'models',
    nntrainer_root / 'nntrainer' / 'graph',
    nntrainer_root / 'nntrainer' / 'compiler',
    nntrainer_root / 'nntrainer' / 'optimizers',
    nntrainer_root / 'nntrainer' / 'tensor',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'fallback',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'cblas_interface',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'x86',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface' / 'nntr_ggml_impl',
    nntrainer_root / 'nntrainer' / 'utils',
    nntrainer_root / 'nntrainer' / 'dataset',
    nntrainer_root / 'nntrainer' / 'schema',
  ]
    nntrainer_inc_args += ['-I' + p]
  endforeach
  nntrainer_abi_cflags = []
endif

nntrainer_dep = declare_dependency(
  dependencies: [libnntrainer, libccapi],
  compile_args: nntrainer_inc_args + nntrainer_abi_cflags,
)

# ── OpenMP / threads ────────────────────────────────────────────────────
openmp_dep = dependency('openmp', required: false)
thread_dep = dependency('threads')

# ── Android log library ─────────────────────────────────────────────────
if is_android
  log_dep = cxx.find_library('log', required: true)
else
  log_dep = dependency('', required: false)
endif

# ── CausalLM includes ──────────────────────────────────────────────────
causallm_auto_inc_dirs = []
foreach _causallm_inc_root : [
  causallm_root / 'layers',
  causallm_root / 'models',
]
  if fs.exists(_causallm_inc_root)
    _found_inc_dirs = run_command(
      'find', _causallm_inc_root, '-type', 'd',
      check: true,
    ).stdout().strip()
    if _found_inc_dirs != ''
      causallm_auto_inc_dirs += _found_inc_dirs.split('\n')
    endif
  endif
endforeach

causallm_inc_args = ['-I' + causallm_root]
foreach d : causallm_auto_inc_dirs
  causallm_inc_args += ['-I' + d]
endforeach

# minja / nlohmann-json (chat template support)
causallm_inc_args += ['-I' + causallm_root / 'third_party' / 'minja' / 'include']
causallm_inc_args += ['-I' + causallm_root / 'third_party']

# ── XGrammar (folded into libcausallm.so; see below) ─────────────────────
causallm_inc_args += ['-I' + causallm_root / 'api']
causallm_inc_args += ['-I' + causallm_root / 'xgrammar' / 'include']
causallm_inc_args += ['-I' + causallm_root / 'xgrammar' / '3rdparty' / 'picojson']
causallm_inc_args += ['-I' + causallm_root / 'xgrammar' / '3rdparty' / 'dlpack' / 'include']

# ── QNN shared infra + public gemma4-e2b-qnn model (gated) ──────────────
if enable_qnn
  causallm_inc_args += ['-I' + causallm_root / 'qnn']
  causallm_inc_args += ['-I' + causallm_root / 'qnn' / 'gemma4-e2b-qnn']
endif

# ── Tokenizer: platform-specific ────────────────────────────────────────
if is_android
  tokenizer_path = causallm_root / 'lib' / 'libtokenizers_android_c.a'
else
  tokenizer_path = causallm_root / 'lib' / 'libtokenizers_c.a'
endif

if not fs.exists(tokenizer_path)
  error('CausalLM tokenizer library not found: @0@'.format(tokenizer_path))
endif

# ── libcausallm.so sources ────────────────────────────────────────────────
# NOTE: Do NOT include causallm_root / 'api' sources here (besides streamer) -
# the API layer is provided by libquick_dot_ai_api.so below; compiling it
# again here would duplicate/interpose symbols.
#
# Keep this list explicit. The previous recursive find silently changed the
# Android ABI whenever an unrelated .cpp file was added under layers/ or
# models/. This is the canonical Android core source manifest.
causallm_src = [
  causallm_root / 'chat_template.cpp',
  causallm_root / 'factory.cpp',
  causallm_root / 'huggingface_tokenizer.cpp',
  causallm_root / 'kv_cache_manager.cpp',
  causallm_root / 'llm_util.cpp',
  causallm_root / 'models' / 'causal_lm.cpp',
  causallm_root / 'models' / 'transformer.cpp',
  causallm_root / 'models' / 'sentence_transformer.cpp',
  causallm_root / 'models' / 'qwen2' / 'qwen2_causallm.cpp',
  causallm_root / 'models' / 'qwen2' / 'qwen2_embedding.cpp',
  causallm_root / 'models' / 'qwen3' / 'qwen3_causallm.cpp',
  causallm_root / 'models' / 'qwen3' / 'qwen3_embedding.cpp',
  causallm_root / 'models' / 'qwen3_moe' / 'qwen3_moe_causallm.cpp',
  causallm_root / 'models' / 'qwen3_moe' / 'qwen_moe_layer.cpp',
  causallm_root / 'models' / 'qwen3_slim_moe' / 'qwen3_slim_moe_causallm.cpp',
  causallm_root / 'models' / 'qwen3_slim_moe' / 'qwen_moe_layer_fsu.cpp',
  causallm_root / 'models' / 'qwen3_cached_slim_moe' / 'qwen3_cached_slim_moe_causallm.cpp',
  causallm_root / 'models' / 'qwen3_cached_slim_moe' / 'qwen_moe_layer_cached.cpp',
  causallm_root / 'models' / 'gpt_oss' / 'gptoss_causallm.cpp',
  causallm_root / 'models' / 'gpt_oss' / 'gpt_oss_moe_layer.cpp',
  causallm_root / 'models' / 'gpt_oss_cached_slim' / 'gptoss_cached_slim_causallm.cpp',
  causallm_root / 'models' / 'gpt_oss_cached_slim' / 'gpt_oss_moe_layer_cached.cpp',
  causallm_root / 'models' / 'gemma3' / 'gemma3_causallm.cpp',
  causallm_root / 'models' / 'gemma3' / 'embedding_gemma.cpp',
  causallm_root / 'models' / 'gemma3' / 'function.cpp',
  causallm_root / 'models' / 'gemma4' / 'gemma4_causallm.cpp',
  causallm_root / 'models' / 'lfm2' / 'lfm2_causallm.cpp',
  causallm_root / 'models' / 'timm_vit' / 'timm_vit_transformer.cpp',
  causallm_root / 'models' / 'deberta_v2' / 'deberta_v2.cpp',
  causallm_root / 'models' / 'bert' / 'bert_transformer.cpp',
  causallm_root / 'models' / 'bert' / 'multilingual_tinybert_16mb.cpp',
  causallm_root / 'models' / 'xlm_roberta' / 'xlm_roberta.cpp',
  causallm_root / 'layers' / 'embedding_layer.cpp',
  causallm_root / 'layers' / 'embedding_pooling_layer.cpp',
  causallm_root / 'layers' / 'embedding_normalize_layer.cpp',
  causallm_root / 'layers' / 'per_layer_slice.cpp',
  causallm_root / 'layers' / 'scalar_multiply.cpp',
  causallm_root / 'layers' / 'logit_softcapping.cpp',
  causallm_root / 'layers' / 'mha_core.cpp',
  causallm_root / 'layers' / 'lm_head.cpp',
  causallm_root / 'layers' / 'reshaped_rms_norm.cpp',
  causallm_root / 'layers' / 'custom_multiply.cpp',
  causallm_root / 'layers' / 'causal_conv1d_layer.cpp',
  causallm_root / 'layers' / 'rms_norm.cpp',
  causallm_root / 'layers' / 'rms_reverse_norm.cpp',
  causallm_root / 'layers' / 'swiglu.cpp',
  causallm_root / 'layers' / 'tie_word_embedding.cpp',
  causallm_root / 'layers' / 'qkv_layer.cpp',
  causallm_root / 'layers' / 'deberta_attention_layer.cpp',
  causallm_root / 'layers' / 'shared_fully_connected_layer.cpp',
  causallm_root / 'api' / 'streamer.cpp',
]

causallm_core_cpp_args = []
quick_dot_ai_api_cpp_args = []
quick_dot_ai_test_cpp_args = []

# Single QNN switch. Keep the QNN ABI-affecting definitions on the targets
# that consume them instead of applying them to the whole Meson project.
if enable_qnn
  causallm_core_cpp_args += [
    '-DENABLE_QNN=1',
    '-DENABLE_QNN_MODELS=1',
  ]
  quick_dot_ai_api_cpp_args += [
    '-DENABLE_QNN_MODELS=1',
    '-DQUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API=1',
  ]
  quick_dot_ai_test_cpp_args += [
    '-DENABLE_QNN=1',
    '-DENABLE_QNN_MODELS=1',
  ]
endif

# ── QNN model-runtime infra + public gemma4-e2b-qnn model (gated) ────────
# quick_dot_ai_qnn.cpp registers the QNN backend at runtime
# (Engine::Global().registerContext("libqnn_context.so", ...)) rather than
# linking the QNN SDK directly, so no QNN SDK headers are needed to compile
# these sources.
causallm_qnn_src = []
if enable_qnn
  causallm_qnn_src += [
    causallm_root / 'qnn' / 'android_memory_allocator.cpp',
    causallm_root / 'qnn' / 'quick_dot_ai_qnn.cpp',
    causallm_root / 'qnn' / 'graph_parser.cpp',
    causallm_root / 'qnn' / 'generate_qnn_utils.cpp',
    causallm_root / 'qnn' / 'qnn_kv_cache_manager.cpp',
    causallm_root / 'qnn' / 'gemma4-e2b-qnn' / 'gemma4_e2b_qnn.cpp',
  ]
endif

# ── XGrammar (wrapper + vendored library), compiled into libcausallm.so ──
# api/xgrammar_manager.cpp defines a Meyers singleton (XGrammarManager::
# Instance()); compiling it into more than one shared library risks two live
# instances. Build it once here as a static library and link_whole it into
# libcausallm.so, so libquick_dot_ai_api.so borrows the single definition at
# runtime via causallm_dep. CPU-only builds get xgrammar too - it isn't
# QNN-specific (mirrors Applications/CausalLM/meson.build, S1).
#
# NOTE: xgrammar_wrapper.h does `#include <transformer.h>` (angle-bracket, so
# it resolves via -I, not a relative path) - transformer.h lives directly in
# causallm_root/models, which is one of the auto-globbed causallm_inc_args
# entries below. Reuse the full causallm_inc_args here rather than a
# hand-picked subset.
causallm_xgrammar_inc_args = causallm_inc_args

causallm_xgrammar_src = [
  # wrapper: bridges causallm::XGrammar to the xgrammar library.
  causallm_root / 'api' / 'xgrammar_wrapper.cpp',
  causallm_root / 'api' / 'xgrammar_manager.cpp',
  # xgrammar library sources (vendored xgrammar submodule checkout).
  causallm_root / 'xgrammar' / 'cpp' / 'compiled_grammar.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'config.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'earley_parser.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'fsm_builder.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'fsm.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_builder.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_compiler.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_functor.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_matcher.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_parser.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar_printer.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'grammar.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'json_schema_converter_ext.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'json_schema_converter.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'regex_converter.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'structural_tag.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'tokenizer_info.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'testing.cc',
  # Support files (required for linking)
  causallm_root / 'xgrammar' / 'cpp' / 'support' / 'logging.cc',
  causallm_root / 'xgrammar' / 'cpp' / 'support' / 'recursion_guard.cc',
]

causallm_xgrammar_lib = static_library('causallm_xgrammar',
  causallm_xgrammar_src,
  dependencies: [nntrainer_dep],
  cpp_args: causallm_xgrammar_inc_args,
  # The vendored xgrammar sources are not warning-clean under this project's
  # stricter default warning level; relax werror for this static library
  # only (mirrors Applications/CausalLM/meson.build, S1).
  override_options: ['werror=false'],
)

# ── Build libcausallm.so ─────────────────────────────────────────────────
causallm_lib = shared_library('causallm',
    causallm_src,
    causallm_qnn_src,
    dependencies: [nntrainer_dep, openmp_dep, log_dep],
    cpp_args: causallm_inc_args + causallm_core_cpp_args,
    link_args: [tokenizer_path],
    link_whole: [causallm_xgrammar_lib],
    install: true,
)

causallm_dep = declare_dependency(
    link_with: causallm_lib,
    compile_args: causallm_inc_args,
)

# ── libquick_dot_ai_api.so (+ quick_dot_ai_test) ─────────────────────────
# Mirrors QuickAI's api/meson.build: xgrammar (and, when enabled,
# QNN model sources) are compiled exactly once into libcausallm.so above;
# this target borrows those symbols via causallm_dep instead of compiling its
# own copies. ENABLE_QNN_MODELS is defined only when enable-qnn is set,
# matching the QNN sources conditionally compiled into libcausallm.so - the
# dynamic_cast<Quick_Dot_AI_QNN *> / make_unique<Gemma4_E2B_QNN> paths that
# define unlocks in api/quick_dot_ai_api.cpp then resolve at link time via
# causallm_dep.
if get_option('enable-api')
  quick_dot_ai_inc_args = [
    '-I' + causallm_root / 'api',
    '-I' + causallm_root / 'xgrammar' / 'include',
    '-I' + causallm_root / 'xgrammar' / '3rdparty' / 'picojson',
    '-I' + causallm_root / 'xgrammar' / '3rdparty' / 'dlpack' / 'include',
  ]

  if enable_qnn
    quick_dot_ai_inc_args += [
      '-I' + causallm_root / 'qnn',
      '-I' + causallm_root / 'qnn' / 'gemma4-e2b-qnn',
    ]
  endif

  quick_dot_ai_api_lib = shared_library('quick_dot_ai_api',
      causallm_root / 'api' / 'quick_dot_ai_api.cpp',
      causallm_root / 'api' / 'model_callbacks.cpp',
      causallm_root / 'api' / 'model_config.cpp',
      causallm_root / 'api' / 'model_descriptors_public.cpp',
      causallm_root / 'api' / 'streamer.cpp',
      causallm_root / 'api' / 'callback_streamer.cpp',
      dependencies: [nntrainer_dep, causallm_dep, openmp_dep, thread_dep, log_dep],
      cpp_args: quick_dot_ai_inc_args + quick_dot_ai_api_cpp_args,
      install: true,
  )

  quick_dot_ai_api_dep = declare_dependency(
      link_with: quick_dot_ai_api_lib,
      compile_args: ['-I' + causallm_root / 'api'],
      dependencies: [causallm_dep],
  )
endif

# ── quick_dot_ai_test executable ─────────────────────────────────────────
if get_option('enable-api-test')
  if not get_option('enable-api')
    error('enable-api-test requires enable-api')
  endif

  executable('quick_dot_ai_test',
      causallm_root / 'api-app' / 'test_api.cpp',
      dependencies: [quick_dot_ai_api_dep],
      cpp_args: quick_dot_ai_test_cpp_args,
      install: false,
  )
endif

# ── nntr_causallm: simple path-based CLI (main.cpp) ──────────────────────
# Unlike quick_dot_ai_test (which takes a catalog model *name* via the C API),
# this links libcausallm directly and takes a filesystem <model_path> [prompt],
# loading config.json / nntr_config.json from that directory (mirrors the
# in-tree Applications/CausalLM/meson.build target). Public CPU models only —
# the architectures main.cpp registers; gauss/QNN models are not registered here.
executable('nntr_causallm',
    causallm_root / 'main.cpp',
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, thread_dep, log_dep],
    cpp_args: causallm_inc_args,
    install: false,
)

# ── Native tools ─────────────────────────────────────────────────────────
executable('nntr_quantize',
    causallm_root / 'quantize.cpp',
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, thread_dep, log_dep],
    cpp_args: causallm_inc_args,
    install: false,
)

executable('nntr_safetensors_info',
    causallm_root / 'safetensors_info.cpp',
    dependencies: [nntrainer_dep],
    cpp_args: causallm_inc_args,
    install: false,
)
