cl_op_sources = [
  'blas_kernels.cpp',
  'blas_kernel_interface.cpp',
  'attention_kernel_interface.cpp',
  'attention_kernels.cpp',
  'clblast_interface.cpp',
  'cl_compute_ops.cpp'
]

cl_op_headers = [
  'blas_kernel_interface.h',
  'attention_kernel_interface.h',
  'clblast_interface.h'
]

if get_option('enable-fp16')
  cl_op_sources += 'blas_kernels_fp16.cpp'
  cl_op_sources += 'attention_kernels_fp16.cpp'
endif

subdir('cl_kernels')

foreach s : cl_op_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : cl_op_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
