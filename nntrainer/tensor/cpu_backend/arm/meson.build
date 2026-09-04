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


# cpu_backend.h declares hgemm.h's flash-attention GEMM kernels (e.g.
# hgemm_f16xf16_f32_fmlal) outside ENABLE_FP16 so ARM+NEON translation
# units built with ENABLE_FP16=0 can still see them and resolve the
# definitions at link time against a library built with enable-fp16=true.
# Keep hgemm.h on the include path regardless of enable-fp16; only the
# kernel sources themselves stay gated below.
nntrainer_inc += include_directories('hgemm')
nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'

if get_option('enable-fp16')
  arm_compute_backend_sources += 'arm_compute_backend_fp16.cpp'
  arm_compute_backend_sources += 'neon_impl_fp16.cpp'

  subdir('hgemm')
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
