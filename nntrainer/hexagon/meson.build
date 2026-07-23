hexagon_headers = [
  'hexagon_context.h',
  'hexagon_compute_ops.h',
]

nntrainer_sources += meson.current_source_dir() / 'hexagon_context.cpp'
nntrainer_sources += meson.current_source_dir() / 'hexagon_compute_ops.cpp'

foreach h : hexagon_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
