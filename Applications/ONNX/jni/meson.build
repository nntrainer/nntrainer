onnx_example_sources = [
  'main.cpp',
]

onnx_example_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  onnx_example_dependencies += [gtest_dep]
endif

e = executable('nntrainer_onnx_example',
  onnx_example_sources,
  include_directories: [include_directories('.')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
