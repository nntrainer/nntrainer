pal_headers = [
  'include/PAL/Debug.hpp',
  'include/PAL/Directory.hpp',
  'include/PAL/DynamicLoading.hpp',
  'include/PAL/FileOp.hpp',
  'include/PAL/GetOpt.hpp',
  'include/PAL/Path.hpp',
  'include/PAL/StringOp.hpp'
]

pal_sources = [
  'src/common/GetOpt.cpp',
  'src/common/StringOp.cpp',
  'src/linux/Directory.cpp',
  'src/linux/DynamicLoading.cpp',
  'src/linux/FileOp.cpp',
  'src/linux/Path.cpp'
]

foreach s : pal_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

nntrainer_inc_abs += meson.current_source_dir() / 'include'

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('PAL', install_dir : qnn_include_dir )

install_headers(pal_headers, install_dir : qnn_include_dir / 'PAL')