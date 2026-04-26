nntrainer_inc += include_directories('.')


wrapper_util_sources = [
  'QnnWrapperUtils.cpp'
]

wrapper_util_headers=[
  'QnnWrapperUtils.hpp'
]

foreach s : wrapper_util_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('WrapperUtils', install_dir : qnn_include_dir )

install_headers(wrapper_util_headers, install_dir : qnn_include_dir / 'WrapperUtils')