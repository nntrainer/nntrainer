# SPDX-License-Identifier: Apache-2.0
# YOLOv11m detection inference example

yolov11_infer_src = [
  meson.current_source_dir() / 'main.cpp',
  meson.current_source_dir() / 'c2psa_layer.cpp',
]

yolov11_inc = include_directories('.')

# Optional direct image (.jpg/.png) input: enabled only when stb_image.h is
# present in this directory (it is not committed). Without it the example still
# builds and runs on raw .bin input, so CI stays green.
#   curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
#     -o Applications/CausalLM/models/YOLOv11/jni/stb_image.h
yolov11_cpp_args = []
if import('fs').exists(meson.current_source_dir() / 'stb_image.h')
  yolov11_cpp_args += '-DYOLO_WITH_STB_IMAGE'
  message('YOLOv11: stb_image.h found -> direct image input enabled')
endif

yolov11_infer = executable('yolov11_infer',
  yolov11_infer_src,
  include_directories: [yolov11_inc],
  cpp_args: yolov11_cpp_args,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  build_by_default: get_option('enable-app'),
  install: false,
)
