backprop_train_sources = [
  'backprop_train.cpp',
  'mnist_loader.cpp',
]

lora_train_sources = [
  'lora_train.cpp',
  'mnist_loader.cpp',
]

inference_sources = [
  'inference.cpp',
  'mnist_loader.cpp',
]

lora_example_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  lora_example_dependencies += [gtest_dep]
endif

e = executable('backprop_train',
  backprop_train_sources,
  include_directories: [include_directories('.')],
  dependencies: lora_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

e2 = executable('lora_train',
  lora_train_sources,
  include_directories: [include_directories('.')],
  dependencies: lora_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# Inference-only executable
e3 = executable('inference',
  inference_sources,
  include_directories: [include_directories('.')],
  dependencies: lora_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
