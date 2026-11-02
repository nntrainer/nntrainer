# Shared by the in-tree and prebuilt-engine projects. Source paths stay as
# strings so the standalone project can reference files outside its root.
causallm_htp_model_src = [
  causallm_root / 'models' / 'gemma4' / 'gemma4_htp_causallm.cpp',
  causallm_root / 'models' / 'gemma4' / 'gemma4_htp_moe_causallm.cpp',
  causallm_root / 'models' / 'htp_weight_preparation.cpp',
  causallm_root / 'models' / 'lfm2_moe' / 'lfm2_moe_layer.cpp',
  causallm_root / 'models' / 'lfm2_moe' / 'lfm2_htp_moe_causallm.cpp',
]

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

causallm_qnn_src = []
if causallm_enable_qnn
  causallm_qnn_src += [
    causallm_root / 'qnn' / 'android_memory_allocator.cpp',
    causallm_root / 'qnn' / 'quick_dot_ai_qnn.cpp',
    causallm_root / 'qnn' / 'graph_parser.cpp',
    causallm_root / 'qnn' / 'generate_qnn_utils.cpp',
    causallm_root / 'qnn' / 'qnn_kv_cache_manager.cpp',
    causallm_root / 'qnn' / 'gemma4-e2b-qnn' / 'gemma4_e2b_qnn.cpp',
  ]
endif

causallm_common_cpp_args = []

# Probe the archive each entry point actually links, with its platform libs.
tokenizer_cache = false
if not causallm_tokenizer_cache_opt.disabled()
  tokenizer_cache = cxx.has_function('tokenizers_new_from_snapshot',
    args: causallm_tokenizer_probe_args,
    dependencies: causallm_tokenizer_probe_deps)
  if not tokenizer_cache and causallm_tokenizer_cache_opt.enabled()
    error('@0@ is enabled, but @1@ does not export the snapshot entry points. Rebuild it with Applications/CausalLM/tools/build_tokenizers_c.sh.'.format(
      causallm_tokenizer_cache_name, causallm_tokenizer_probe_args[0]))
  endif
endif
if not tokenizer_cache
  causallm_common_cpp_args += ['-DCAUSALLM_DISABLE_TOKENIZER_CACHE']
endif
if causallm_enable_qnn
  causallm_common_cpp_args += ['-DENABLE_QNN=1', '-DENABLE_QNN_MODELS=1']
endif
