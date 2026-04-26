utils_sources = [
  'DataUtil.cpp',
  'DynamicLoadUtil.cpp',
  'IOTensor.cpp',
  'QnnSampleAppUtils.cpp'
]

utils_headers = [
  'BuildId.hpp',
  'DataUtil.hpp',
  'DynamicLoadUtil.hpp',
  'IOTensor.hpp',
  'QnnSampleAppUtils.hpp'
]

foreach s : utils_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('Utils', install_dir : qnn_include_dir )

install_headers(utils_headers, install_dir : qnn_include_dir / 'Utils')