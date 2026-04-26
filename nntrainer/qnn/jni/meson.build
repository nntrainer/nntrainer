# nntrainer's own QNN integration plugin. The Qualcomm QNN SDK headers
# are pulled in via -Dqnn-sdk-root (see ../meson.build). The proprietary
# Qualcomm sample utility code that previously lived in qnn/{Log,PAL,Utils,
# WrapperUtils} and qnn-api/ is NOT shipped — users must rely on the SDK
# tree they obtain from Qualcomm. See docs/backend_guide/QNN_BUILD.md.
#
# The plugin is built as a separate shared library (libqnn_context.so) so
# that the QNN SDK dependency stays out of the main libnntrainer.

qnn_context_sources = [
  meson.current_source_dir() / 'qnn_rpc_manager.cpp',
  meson.current_source_dir() / '..' / '..' / 'qnn_context.cpp',
]

qnn_context_headers = [
  meson.current_source_dir() / 'qnn_rpc_manager.h',
  meson.current_source_dir() / 'qnn_context_var.h',
  meson.current_source_dir() / 'iotensor_wrapper.hpp',
]

shared_library('qnn_context',
  qnn_context_sources,
  dependencies: nntrainer_dep,
  include_directories: nntrainer_inc,
  cpp_args: ['-DPLUGGABLE'],
  install: true,
  install_dir: nntrainer_libdir,
)

# Make the kept nntrainer-side QNN headers reachable from elsewhere.
nntrainer_inc += include_directories('.')
nntrainer_inc_abs += meson.current_source_dir()

# Layer-level QNN integration (QNNGraph / QNNLinear) and shared properties.
subdir('qnn')
