# QNN context is built as a plugin shared library (libqnn_context.so).
# It is loaded dynamically by Engine::registerContext("libqnn_context.so").
# This decouples the QNN SDK dependency from the main nntrainer library.

qnn_context_sources = [
  meson.current_source_dir() / 'qnn_rpc_manager.cpp',
  meson.current_source_dir() / '..' / '..' / 'qnn_context.cpp',
]

qnn_context_headers = [
  meson.current_source_dir() / 'qnn_rpc_manager.h',
  meson.current_source_dir() / 'qnn_context_var.h',
]

# Build as shared library plugin
shared_library('qnn_context',
  qnn_context_sources,
  dependencies: nntrainer_dep,
  include_directories: nntrainer_inc,
  cpp_args: ['-DPLUGGABLE'],
  install: true,
  install_dir: nntrainer_libdir,
)

# Include directories for QNN
nntrainer_inc += include_directories('.')
nntrainer_inc_abs += meson.current_source_dir()

# Add include paths so #include <IOTensor.hpp> works
nntrainer_inc += include_directories('qnn', 'qnn/Utils', 'qnn/Log', 'qnn/PAL/include')
nntrainer_inc_abs += meson.current_source_dir() / 'qnn'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'Utils'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'Log'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'PAL' / 'include'

# Include QNN subdirectories
subdir('qnn')
subdir('qnn-api')
