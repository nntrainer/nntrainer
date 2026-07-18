opencl_sources = [
    'opencl_buffer.cpp',
    'opencl_buffer_manager.cpp',
    'opencl_command_queue_manager.cpp',
    'opencl_context_manager.cpp',
    'opencl_device_info.cpp',
    'opencl_kernel.cpp',
    'opencl_loader.cpp',
    'opencl_program.cpp'
]

opencl_headers = [
  'CL/cl_platform.h',
  'opencl_buffer.h',
  'opencl_buffer_manager.h',
  'opencl_command_queue_manager.h',
  'opencl_context_manager.h',
  'opencl_kernel.h',
  'opencl_loader.h',
  'opencl_device_info.h',
  'opencl_program.h'
]

foreach s : opencl_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : opencl_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

install_subdir('CL', install_dir : nntrainer_prefix / 'include' / 'nntrainer')
