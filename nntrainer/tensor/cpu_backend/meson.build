cpu_backend_headers = [
  'cpu_backend.h',
  'compute_ops.h',
]

nntrainer_sources += meson.current_source_dir() / 'compute_ops.cpp'
nntrainer_sources += meson.current_source_dir() / 'cpu_ops_table.cpp'

subdir('fallback')
nntrainer_inc += include_directories('fallback')
nntrainer_inc_abs += meson.current_source_dir() / 'fallback'

subdir('ggml_interface')
nntrainer_inc += include_directories('ggml_interface')
nntrainer_inc_abs += meson.current_source_dir() / 'ggml_interface'

if get_option('enable-blas')
  subdir('cblas_interface')
  nntrainer_inc += include_directories('cblas_interface')
  nntrainer_inc_abs += meson.current_source_dir() / 'cblas_interface'
endif

arch = host_machine.cpu_family()
if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
    subdir('arm')
    nntrainer_inc += include_directories('arm')
    nntrainer_inc_abs += meson.current_source_dir() / 'arm'
elif arch == 'x86_64' or arch == 'x86'
    subdir('x86')
    nntrainer_inc += include_directories('x86')
    nntrainer_inc_abs += meson.current_source_dir() / 'x86'
endif

foreach h : cpu_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
