cuda_sources = [
  'cuda_context_manager.cpp',
  'cuda_util.cpp',
]

cuda_headers = [
  'cuda_context_manager.h',
  'cuda_util.h',
]

foreach s : cuda_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : cuda_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
