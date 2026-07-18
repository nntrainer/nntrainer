# Param-general LLM layers promoted from Applications/CausalLM/layers/.
# Compiled straight into libnntrainer (the loss/ subdir pattern), NOT per-layer
# shared_library plugins. Add sources incrementally, one per promotion.
llm_layer_sources = [
  'logit_softcapping.cpp',
  'scalar_multiply.cpp',
  'qkv_layer.cpp',
  'geglu_layer.cpp',
  'swiglu_layer.cpp',
  'lm_head.cpp', # untied host lm_head FC (no OpenCL)
  # tie_word_embedding's OpenCL lm_head GEMV is #if defined(ENABLE_OPENCL)-guarded
  # internally, so it compiles host-only without OpenCL (the FP32 unittest build).
  'tie_word_embedding.cpp',
]

llm_layer_headers = [
  'layer_prof.h', # env-gated per-layer profiler (header-only)
  'lm_head.h',
  'logit_softcapping.h',
  'scalar_multiply.h',
  'scalar_multiply_gpu.h',
  'tie_word_embedding.h',
  'qkv_layer.h',
  'geglu_layer.h',
  'swiglu_layer.h',
]

# scalar_multiply's GPU variant uses OpenCL directly (cl_mem / ClContext), so it
# compiles only when the OpenCL backend is on — mirroring the cl_context.cpp
# registration that consumes it (a CPU-only libnntrainer must not pull it in).
if get_option('enable-opencl')
  llm_layer_sources += 'scalar_multiply_gpu.cpp'
endif

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : llm_layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
