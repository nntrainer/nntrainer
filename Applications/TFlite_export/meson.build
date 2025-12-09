#copy ini file to build/res directory
app_res_dir = meson.current_source_dir() / 'res'
build_app_res_dir = meson.current_build_dir()
run_command(['cp', '-r', app_res_dir, build_app_res_dir], check: true)

app_sources = [
  'main.cpp',
]

app_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_tflite_export_example',
  app_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: app_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
