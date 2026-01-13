cuda_sources = [
  'cuda_context_manager.cpp',
  'cuda_util.cpp',
  'custom_dot_wrapper_cuda.cpp',
]

cuda_headers = [
  'cuda_context_manager.h',
  'cuda_util.h',
  'custom_dot_wrapper_cuda.h',
  'gemm_int4_cuda.h',
  'dequantize_cuda.h',
  'quantize_cuda.h',
]

foreach s : cuda_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

if get_option('enable-cuda')
  cuda_sources_cu = [
    'gemm_int4_cuda.cu',
    'quantize_cuda.cu',
    'dequantize_cuda.cu'
  ]

  nvcc = find_program('nvcc', required: true)
  if nvcc.found()
    foreach src : cuda_sources_cu
      obj_name = src.replace('.cu', '.o')
      # Basic flags, adjust architecture as needed
      cuda_flags = ['-std=c++20', '-gencode', 'arch=compute_89,code=sm_89', '-gencode', 'arch=compute_120,code=sm_120', '-c', '--compiler-options', '-fPIC']
      if host_machine.system() == 'windows'
        cuda_flags = ['-std=c++20', '-gencode', 'arch=compute_89,code=sm_89', '-gencode', 'arch=compute_120,code=sm_120', '-c']
        cuda_flags += ['-Xcompiler', '/MD']
        cuda_flags += ['-Xcompiler', '/wd4819']
      endif

      obj = custom_target(obj_name,
        command: [nvcc] + cuda_flags + ['@INPUT@', '-o', '@OUTPUT@'] + ['-I@0@'.format(meson.source_root() / 'nntrainer')],
        input: src,
        output: obj_name
      )
      nntrainer_sources += obj
    endforeach
  endif
endif

foreach h : cuda_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
