arm_compute_backend_headers = [
    'arm_compute_backend.h',
    'neon_impl.h',
    'neon_setting.h',
    'neon_mathfun.h',
    'neon_mathfun.hxx',
]
arm_compute_backend_sources = [
    'arm_compute_backend.cpp',
    'neon_impl.cpp',
]


subdir('matrix_transpose_neon')
nntrainer_inc += include_directories('matrix_transpose_neon')
nntrainer_inc_abs += meson.current_source_dir() / 'matrix_transpose_neon'


if get_option('enable-fp16')
  arm_compute_backend_sources += 'arm_compute_backend_fp16.cpp'
  arm_compute_backend_sources += 'neon_impl_fp16.cpp'
  arm_compute_backend_sources += 'kleidiai_interface_qai8dxp_qsi4cxp.cpp'
  arm_compute_backend_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p.cpp'
  
  # Select thread backend for kleidiai parallel implementation
  if get_option('thread-backend') == 'bstp'
    arm_compute_backend_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_stdthread.cpp'
  elif get_option('thread-backend') == 'omp'
    arm_compute_backend_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
  elif get_option('thread-backend') == 'mixed'
    arm_compute_backend_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
  else
    message('Unknown thread-backend meson option! Use omp as a default for kleidiai.')
    arm_compute_backend_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
  endif
  
  arm_compute_backend_headers += 'kleidiai_interface.h'

  subdir('hgemm')
  nntrainer_inc += include_directories('hgemm')
  nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'

  subdir('kai')
  nntrainer_inc += include_directories('kai')
  nntrainer_inc_abs += meson.current_source_dir() / 'kai'
endif

arch = host_machine.cpu_family()
if arch == 'arm'
  arm_compute_backend_headers += 'armv7_neon.h'
endif

foreach s : arm_compute_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : arm_compute_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
