# Find CUDA compiler
dep = dependency('cuda', version : '>=13', modules : ['cublas'])

nvcc = find_program('nvcc', required: true)

if nvcc.found()
  cuda_sources = [
    'rmsnorm_cuda.cu',
    'addition_cuda.cu'
  ]

  cuda_headers = [
    'rmsnorm_cuda.h',
    'addition_cuda.h',
    'cuda_interface.h'
  ]

  kernel_objects = []
  foreach kernel : cuda_sources
    obj_name = kernel.replace('.cu', '.o')
    obj = custom_target(obj_name,
      command: [nvcc, '-c', '-Xcompiler', '/MD', '@INPUT@', '-o', '@OUTPUT@'],
      input: kernel,
      output: obj_name
    )
    kernel_objects += obj
  endforeach

  # Add cuda_interface.cpp to regular sources
  nntrainer_sources += meson.current_source_dir() / 'cuda_interface.cpp'

  nntrainer_sources += kernel_objects

  foreach h : cuda_headers
    nntrainer_headers += meson.current_source_dir() / h
  endforeach

else
  message('CUDA compiler (nvcc) not found. CUDA kernels will not be compiled.')
endif
