hexagon_host_sources = [
  'host/rpcmem_allocator.cpp',
  'host/hexagon_runner.cpp',
  'host/graph_lowering.cpp',
]

hexagon_host_headers = [
  'host/rpcmem_allocator.h',
  'host/hexagon_runner.h',
  'host/graph_lowering.h',
  'htp/nntr_htp_common.h',
]

foreach s : hexagon_host_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hexagon_host_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

# QAIC-generated FastRPC stub (hvx_gen_dir is set in the top-level
# meson.build enable-hexagon block and is in meson's flat scope here).
nntrainer_sources += hvx_gen_dir / 'nntr_htp_stub.c'

nntrainer_inc += include_directories('host', 'htp')
nntrainer_inc_abs += meson.current_source_dir() / 'host'
nntrainer_inc_abs += meson.current_source_dir() / 'htp'
nntrainer_inc_abs += hvx_gen_dir
nntrainer_inc_abs += hvx_sdk_root / 'incs'
nntrainer_inc_abs += hvx_sdk_root / 'incs' / 'stddef'
nntrainer_inc_abs += hvx_sdk_root / 'ipc' / 'fastrpc' / 'rpcmem' / 'inc'
