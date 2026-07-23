# SPDX-License-Identifier: Apache-2.0
# YOLOv7ReIDtiny pose + ReID inference example

yolov7_pose_infer_src = [
  meson.current_source_dir() / 'main.cpp',
]

yolov7_pose_inc = include_directories('.')

yolov7_pose_infer = executable('yolov7_pose_infer',
  yolov7_pose_infer_src,
  include_directories: [yolov7_pose_inc, quick_ai_layer_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_rtmcc_head_dep],
  build_by_default: get_option('enable-app'),
  install: false,
)
