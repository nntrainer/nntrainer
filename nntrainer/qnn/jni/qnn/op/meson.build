op_sources=[
  'QNNLinear.cpp',
  'QNNGraph.cpp',
]

op_headers=[
  'QNNLinear.h',
  'QNNGraph.h',
]

foreach s : op_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

qnn_include_dir = nntrainer_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('op', install_dir : qnn_include_dir )

install_headers(op_headers, install_dir : qnn_include_dir / 'op')