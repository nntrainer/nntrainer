# SPDX-License-Identifier: Apache-2.0
# FastViTKeyword classification inference example

fastvit_keyword_infer_src = [
  meson.current_source_dir() / 'main.cpp',
  meson.current_source_dir() / 'fastvit_attention_layer.cpp',
]

fastvit_keyword_inc = include_directories('.')

fastvit_keyword_infer = executable('fastvit_keyword_infer',
  fastvit_keyword_infer_src,
  include_directories: [fastvit_keyword_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  build_by_default: get_option('enable-app'),
  install: false,
)
