build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'

vgg_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]
if get_option('platform') == 'tizen'
  if not gtest_dep.found()
    error('Gtest dependency not found for VGG application')
  endif
endif

executable('nntrainer_vgg',
  vgg_sources,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.'), cifar_include_dir],
  install: get_option('install-app'),
  install_dir: application_install_dir
)
