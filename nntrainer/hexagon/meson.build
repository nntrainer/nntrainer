hexagon_headers = [
  'hexagon_context.h',
  'hexagon_compute_ops.h',
  'hexagon_rpc_allocator.h',
]

nntrainer_sources += meson.current_source_dir() / 'hexagon_context.cpp'
nntrainer_sources += meson.current_source_dir() / 'hexagon_compute_ops.cpp'
nntrainer_sources += meson.current_source_dir() / 'hexagon_rpc_allocator.cpp'

foreach h : hexagon_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
