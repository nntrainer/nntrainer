mixed_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

executable('nntrainer_mixed_example',
  mixed_sources,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.'), cifar_include_dir],
  install: get_option('install-app'),
  install_dir: application_install_dir
)
