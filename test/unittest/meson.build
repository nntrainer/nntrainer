fs = import('fs')

unittest_nntrainer_deps = [
  nntrainer_test_deps,
  nntrainer_ccapi_dep
] # if unittest-wide dep is added, this is the place to add

if get_option('platform') == 'tizen'
  unittest_nntrainer_deps += [ nntrainer_capi_dep ] # so that feature_state can be overridden
endif

# test material need to be unzipped from "(project_home)/packaging/"
unzip_target = [
  ['trainset.tar.gz', ''],
  ['valset.tar.gz', ''],
  ['testset.tar.gz', ''],
  ['unittest_layers.tar.gz', 'unittest_layers'],
  ['unittest_layers_w16a16.tar.gz', 'unittest_layers'],
  ['unittest_layers_embedding.tar.gz', 'unittest_layers'],
  ['unittest_models.tar.gz', 'unittest_models'],
  ['unittest_models_v2.tar.gz', 'unittest_models'],
  ['unittest_models_v3.tar.gz', 'unittest_models'],
  ['unittest_models_multiout.tar.gz', 'unittest_models'],
  ['unittest_models_exeorder.tar.gz', 'unittest_models'],
]

src_path = meson.source_root() / 'packaging'
dest_path = nntrainer_test_resdir

foreach target: unzip_target
  _src_path = src_path / target[0]
  _dest_path = dest_path / target[1]

  if build_machine.system() == 'windows'
    _src_path_win = _src_path.replace('/', '\\')
    _dest_path_win = _dest_path.replace('/', '\\')
    if not fs.exists (_dest_path_win)
      run_command([prog_win_cmd, '/C', 'mkdir', _dest_path_win], check: true)
    endif
    run_command([prog_win_cmd, '/C', 'tar', 'xzf', _src_path_win, '-C', _dest_path_win], check: true)
  else
    run_command(['mkdir', '-p', _dest_path], check: true)
    run_command(['tar', 'xzf', _src_path, '-C', _dest_path], check: true)
  endif
endforeach

src_path_label = src_path / 'label.dat'
dest_path_label = dest_path / 'label.dat'
if build_machine.system() == 'windows'
  src_path_label_win = src_path_label.replace('/', '\\')
  dest_path_label_win = dest_path_label.replace('/', '\\')
  run_command([prog_win_cmd, '/C', 'copy', src_path_label_win, dest_path_label_win], check: true)
else
  run_command(['cp', '-l', src_path_label, dest_path_label], check: true)
endif

test_target = [
  ['unittest_nntrainer_activations', []],
  ['unittest_nntrainer_cpu_backend', []],
  ['unittest_nntrainer_exe_order', []],
  ['unittest_nntrainer_internal', []],
  ['unittest_nntrainer_lazy_tensor', []],
  ['unittest_nntrainer_tensor', []],
  ['unittest_nntrainer_quantizer', []],
  ['unittest_util_func', []],
  ['unittest_nntrainer_models', [
    'models' / 'models_test_utils.cpp', 'models' / 'models_golden_test.cpp'
  ]],
  ['unittest_nntrainer_graph', []],
  ['unittest_base_properties', []],
  ['unittest_common_properties', []],
  ['unittest_nntrainer_tensor_pool', []],
  ['unittest_nntrainer_lr_scheduler', []],
  #['unittest_nntrainer_task', []],
]

if host_machine.system() != 'windows'
  test_target += [['unittest_nntrainer_appcontext', []]]
  test_target += [['unittest_nntrainer_modelfile', []]]
endif

if get_option('enable-opencl')
  test_target += [['unittest_blas_kernels_cl', []]]
  test_target += [['unittest_attention_kernels_cl', []]]
endif

if get_option('enable-cuda')
  test_target += [['unittest_cuda', []]]
endif

if get_option('enable-fp16')
  test_target += [['unittest_nntrainer_tensor_fp16', []]]
  test_target += [['unittest_nntrainer_tensor_pool_fp16', []]]
endif

if get_option('enable-profile')
  if gmock_dep.version().version_compare('>=1.10.0')
    test_target += [['unittest_nntrainer_profiler', []]]
  endif
endif

foreach target: test_target
  cuda_deps = []
  cuda_link_args = []
  if target[0] == 'unittest_cuda' and get_option('enable-cuda')
    cuda_deps = [cuda_dep]
    cuda_link_args = ['-NODEFAULTLIB:LIBCMT', '-NOIMPLIB', '-NOEXP']
  endif
  
  exe = executable(
    target[0],
    [target[0] + '.cpp'] + [target[1]],
    # below is temporary measure, we will eventually remove unittest_nntrainer_models
    include_directories: include_directories('models'),
    dependencies: unittest_nntrainer_deps + cuda_deps,
    link_args: cuda_link_args,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target[0], exe,
    args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), target[0])],
    timeout: test_timeout,
    suite: 'unittests'
  )
endforeach

unittest_inc = include_directories('.')

# subdir('memory')
subdir('compiler')
subdir('layers')
subdir('datasets')
subdir('models')
if get_option('enable-tflite-interpreter')
  subdir('integration_tests')
endif
