# SPDX-License-Identifier: Apache-2.0
# YOLOv7 detection inference example

# Standard YOLOv7 Infer
yolov7_infer_src = [
  meson.current_source_dir() / 'main.cpp',
]

yolov7_inc = include_directories('.', '../../../third_party')

yolov7_infer = executable('yolov7_infer',
  yolov7_infer_src,
  include_directories: [yolov7_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  build_by_default: get_option('enable-app'),
  install: true,
  install_dir: application_install_dir,
)

# YOLOv7-Tiny Infer
yolov7_tiny_infer_src = [
  meson.current_source_dir() / 'main_tiny.cpp',
]

# Include quick_ai root for stb_image.h
yolov7_tiny_inc = include_directories(
  '.',
  '../../../third_party',
  '../../..',  # Applications/quick_ai/ root (for stb_image.h)
)



yolov7_tiny_infer = executable('yolov7_tiny_infer',
  yolov7_tiny_infer_src,
  include_directories: [yolov7_tiny_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  cpp_args: ['-DYOLO_WITH_STB_IMAGE'],
  build_by_default: get_option('enable-app'),
  install: true,
  install_dir: application_install_dir,
)

