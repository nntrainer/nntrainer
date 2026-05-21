# TorchFXConverter model test build
#
# Builds test executables from auto-generated model code (converter.py output).
# Each test links against CausalLM's custom layer libraries and NNTrainer.

converter_test_inc = [
  include_directories('.'),
]

# Core custom layer dependencies (from CausalLM/layers)
converter_layer_deps = [
  causallm_rms_norm_dep,
  causallm_swiglu_dep,
  causallm_mha_core_dep,
  causallm_embedding_layer_dep,
  causallm_reshaped_rms_norm_dep,
  causallm_tie_word_embedding_dep,
]

# Helper: define a converter test executable if its source file exists.
# Each entry: [file_basename, executable_name, class_name]
converter_test_models = [
  # Pre-built Qwen3 (legacy test_model.cpp)
  ['test_model', 'converter_qwen3_test', 'Qwen3CausalLM'],
  # Decoder-only CausalLM
  ['qwen3', 'converter_qwen3_gen_test', 'Qwen3CausalLM'],
  ['llama', 'converter_llama_test', 'LlamaCausalLM'],
  ['qwen3_tied', 'converter_qwen3_tied_test', 'Qwen3CausalLM'],
  ['qwen3_06b', 'converter_qwen3_06b_test', 'Qwen3CausalLM'],
  ['qwen3_17b', 'converter_qwen3_17b_test', 'Qwen3CausalLM'],
  ['granite_40', 'converter_granite_40_test', 'GraniteCausalLM'],
  ['lfm_700m', 'converter_lfm_700m_test', 'Lfm2CausalLM'],
  # Embedding models
  ['qwen3_embedding', 'converter_qwen3_embedding_test', 'Qwen3EmbeddingModel'],
  ['embedding_gemma', 'converter_embedding_gemma_test', 'Gemma2EmbeddingModel'],
  ['kalm_embedding', 'converter_kalm_embedding_test', 'Qwen2EmbeddingModel'],
  # Encoder-only
  ['multilingual_e5', 'converter_multilingual_e5_test', 'XlmRobertaModel'],
  # Custom models
  ['gliner2', 'converter_gliner2_test', 'ExtractorEmbeddingModel'],
  # External KV cache mode
  ['qwen3_ext', 'converter_qwen3_ext_test', 'Qwen3CausalLM'],
]

foreach m : converter_test_models
  _src = meson.current_source_dir() / m[0] + '.cpp'
  if fs.exists(_src)
    executable(m[1],
      [
        meson.current_source_dir() / 'main.cpp',
        _src,
      ],
      include_directories: converter_test_inc,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, converter_layer_deps],
      cpp_args: [
        '-DGENERATED_MODEL_CLASS=' + m[2],
        '-DGENERATED_MODEL_HEADER="' + m[0] + '.h"',
      ],
    )
  endif
endforeach
