res_path = meson.current_source_dir() / '..' / 'res'

run_command(['cp', '-lr', res_path, nntr_app_resdir / 'KNN'], check: true)

knn_sources = [
  'main.cpp'
]

knn_inc = include_directories('.')

e = executable('knn_sample',
  knn_sources,
  dependencies: [app_utils_dep, nntrainer_dep, tflite_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_knn', e, args: [nntr_app_resdir / 'KNN/'])
