multi_inout_sources = [
  'main.cpp',
  'multi_loader.cpp'
]

multi_inout_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  multi_inout_dependencies += [gtest_dep]
endif

e = executable('nntrainer_multi_inout',
  multi_inout_sources,
  include_directories: [include_directories('.')],
  dependencies: multi_inout_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
