multi_input_sources = [
  'main.cpp',
  'multi_loader.cpp'
]

multi_input_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  multi_input_dependencies += [gtest_dep]
endif

e = executable('nntrainer_multi_input',
  multi_input_sources,
  include_directories: [include_directories('.')],
  dependencies: multi_input_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
