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

fs = import('fs')
if fs.is_dir('gauss-3.6-qnn')
  subdir('gauss-3.6-qnn')
endif
if fs.is_dir('gauss-3.8-qnn')
  subdir('gauss-3.8-qnn')
endif
# Multimodal QNN model: gauss-3.8-vit-qnn enabled when
# --enable-experimental-multimodal is set (requires --enable-qnn --qnn-models).
# vjepa2-qnn remains excluded pending adaptation.
if get_option('enable-experimental-multimodal') and fs.is_dir('gauss-3.8-vit-qnn')
  subdir('gauss-3.8-vit-qnn')
endif
if fs.is_dir('gemma4-e2b-qnn')
  subdir('gemma4-e2b-qnn')
endif
# if fs.is_dir('vjepa2-qnn')
#   subdir('vjepa2-qnn')
# endif
if fs.is_dir('gauss-4-qnn')
  subdir('gauss-4-qnn')
endif
if fs.is_dir('gauss4-0.6b-qnn')
  subdir('gauss4-0.6b-qnn')
endif
