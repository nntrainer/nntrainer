cuda_sources = [
  'cuda_context_manager.cpp',
  'cuda_stream_manager.cpp',
  'cuda_buffer.cpp',
  'cuda_buffer_manager.cpp',
  'cuda_module.cpp',
  'cuda_kernel.cpp',
  'cuda_elementwise.cpp',
]

cuda_headers = [
  'cuda_common.h',
  'cuda_context_manager.h',
  'cuda_stream_manager.h',
  'cuda_buffer.h',
  'cuda_buffer_manager.h',
  'cuda_module.h',
  'cuda_kernel.h',
  'cuda_elementwise.h',
]

foreach s : cuda_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : cuda_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
