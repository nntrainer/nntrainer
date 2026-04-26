log_sources = [
  'Logger.cpp',
  'LogUtils.cpp',
]

log_headers = [
  'Logger.hpp',
  'LogUtils.hpp'
]

foreach s : log_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('Log', install_dir : qnn_include_dir )

install_headers(log_headers, install_dir : qnn_include_dir / 'Log')
