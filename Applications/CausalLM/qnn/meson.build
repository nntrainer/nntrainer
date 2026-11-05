qnn_src = [
    meson.current_source_dir() / 'android_memory_allocator.cpp',
    meson.current_source_dir() / 'quick_dot_ai_qnn.cpp',
    meson.current_source_dir() / 'graph_parser.cpp',
    meson.current_source_dir() / 'generate_qnn_utils.cpp',
    meson.current_source_dir() / 'qnn_kv_cache_manager.cpp',
]

qnn_inc = include_directories('.', '../../../api')

quick_dot_ai_src += qnn_src
quick_dot_ai_inc += qnn_inc

# Product QNN models are wired in QuickAI's own src/models/qnn, not here.
fs = import('fs')
if fs.is_dir('gemma4-e2b-qnn')
  subdir('gemma4-e2b-qnn')
endif
# if fs.is_dir('vjepa2-qnn')
#   subdir('vjepa2-qnn')
# endif
