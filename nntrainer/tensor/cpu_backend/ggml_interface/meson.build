subdir('nntr_ggml_impl')
nntrainer_inc += include_directories('nntr_ggml_impl')
nntrainer_inc_abs += meson.current_source_dir() / 'nntr_ggml_impl'

ggml_interface_headers = [
    'ggml_interface.h',
]

# Built for every thread-backend; test/unittest reuses this list.
ggml_interface_common_sources = [
  'ggml_interface.cpp',
  'ggml_interface_q8_0.cpp',
]

ggml_interface_sources = ggml_interface_common_sources

if get_option('thread-backend') == 'bstp'
  ggml_interface_sources += 'ggml_interface_bs_threadpool.cpp'
elif get_option('thread-backend') == 'omp'
  ggml_interface_sources += 'ggml_interface_omp.cpp'
elif get_option('thread-backend') == 'mixed'
  ggml_interface_sources += 'ggml_interface_mixed.cpp'
else
  message('Unknown thread-backend meson option! Use omp as a default.')
  ggml_interface_sources += 'ggml_interface_omp.cpp'
endif

if get_option('enable-fp16')
  ggml_interface_sources += 'ggml_interface_fp16.cpp'
endif

foreach s : ggml_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
