arm_compute_backend_headers = [
    'arm_compute_backend.h',
    'neon_impl.h',
    'neon_mathfun.h',
    'neon_mathfun.hxx',
]
arm_compute_backend_sources = [
    'arm_compute_backend.cpp',
    'neon_impl.cpp',
]

arch = host_machine.cpu_family()
if arch != 'arm'
  subdir('kai_interface')
  nntrainer_inc += include_directories('kai_interface')
  nntrainer_inc_abs += meson.current_source_dir() / 'kai_interface'
endif

subdir('matrix_transpose_neon')
nntrainer_inc += include_directories('matrix_transpose_neon')
nntrainer_inc_abs += meson.current_source_dir() / 'matrix_transpose_neon'


if get_option('enable-fp16')
  arm_compute_backend_sources += 'arm_compute_backend_fp16.cpp'
  arm_compute_backend_sources += 'neon_impl_fp16.cpp'
  
  subdir('hgemm')
  nntrainer_inc += include_directories('hgemm')
  nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'
endif

if arch == 'arm'
  arm_compute_backend_headers += 'armv7_neon.h'
endif

foreach s : arm_compute_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : arm_compute_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
