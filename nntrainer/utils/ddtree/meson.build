ddtree_sources = [
  'ddtree.cpp',
  'ddtree_sliding.cpp',
  'ddtree_compact.cpp',
  'ddtree_sampling.cpp',
]

ddtree_headers = [
  'ddtree_types.h',
  'ddtree.h',
  'ddtree_sliding.h',
  'ddtree_compact.h',
  'ddtree_sampling.h',
]

foreach s : ddtree_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ddtree_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
