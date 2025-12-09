build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'
nntr_cifar_resdir = nntr_app_resdir / 'CIFAR_Classification'
run_command(['cp', '-r', res_path + '/', nntr_cifar_resdir + '/'], check: true)

# prepare default dataset
unzip_target = [
  'trainset.tar.gz',
  'valset.tar.gz',
  'testset.tar.gz'
]

pkg_path = meson.source_root() / 'packaging'

run_command(['mkdir', '-p', nntr_cifar_resdir], check: true)
foreach target: unzip_target
  run_command(['tar', 'xzf', pkg_path / target, '-C', nntr_cifar_resdir], check: true)
endforeach
run_command(['cp', '-l', pkg_path / 'label.dat', nntr_cifar_resdir], check: true)


classification_sources = [
  'main.cpp'
]

e = executable('nntrainer_classification',
  classification_sources,
  dependencies: [app_utils_dep, nntrainer_dep, tflite_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

ini_out_path = nntr_cifar_resdir / 'Classification.ini'

# change epochs to 1
run_command(['sed', '-i', 's/Epochs\ \=\ 30000/Epochs\ \=\ 1/', ini_out_path], check: true)
test('app_cifar_classification', e, args: [ini_out_path, nntr_cifar_resdir])

classification_func_sources = [
  'main_func.cpp'
]

e = executable('nntrainer_classification_func',
  classification_func_sources,
  dependencies: [nntrainer_ccapi_dep, app_utils_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

ini_out_path = nntr_cifar_resdir / 'Classification_func.ini'

# change epochs to 1
run_command(['sed', '-i', 's/Epochs\ \=\ 30000/Epochs\ \=\ 1/', ini_out_path], check: true)

test('app_cifar_classification_func', e, args: [ini_out_path], timeout: 60)
