optimizers_sources= [
    'main.cpp',
    cifar_path / 'cifar_dataloader.cpp'
]

optimizers_dependencies = [app_utils_dep,
  nntrainer_ccapi_dep
]

mnist_ini = configure_file(
  input: '../../MNIST/res/mnist.ini',
  output: 'mnist.ini',
  copy: true
)

mnist_dat = configure_file(
  input: '../../MNIST/res/mnist_trainingSet.dat',
  output: 'mnist_trainingSet.dat',
  copy: true
)

e = executable('nntrainer_optimizers',
  optimizers_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: optimizers_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
