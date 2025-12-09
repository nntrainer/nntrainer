app_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

app_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  app_dependencies += [gtest_dep]
endif

e = executable('nntrainer_simplefc',
  app_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: app_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

if get_option('enable-long-test')
  testenv = environment()
  testenv.set('OPENBLAS_NUM_THREADS', '4')
  test('app_asynch_fsu', e, args: ['fake', '1', '128', '1'], env: testenv, timeout: 300)
endif
