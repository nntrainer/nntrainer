graph_sources = [
  'network_graph.cpp',
  'graph_core.cpp',
  'connection.cpp',
  'checkpoint_block.cpp'
]

graph_headers = [
  'connection.h',
  'graph_node.h',
  'network_graph.h',
  'graph_core.h',
  'graph_node.h',
  'checkpoint_block.h',
]

foreach s : graph_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : graph_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

if get_option('enable-test')
  gradient_checkpointing_test_dir = '..' / '..' / 'test' / 'unittest' / 'gradient_checkpointing'
  nntrainer_inc += include_directories(gradient_checkpointing_test_dir)
  nntrainer_sources += files(join_paths(gradient_checkpointing_test_dir,
                                        'gradient_checkpointing_test_util.cpp'))
endif
