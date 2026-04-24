resnet_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

resnet_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  resnet_dependencies += [gtest_dep]
endif

e = executable('nntrainer_resnet18',
  resnet_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: resnet_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

if get_option('enable-long-test')
  testenv = environment()
  testenv.set('OPENBLAS_NUM_THREADS', '4')
  test('app_resnet18', e, args: ['fake', '1', '128', '1'], env: testenv, timeout: 300)
endif
