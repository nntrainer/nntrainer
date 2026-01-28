# CUDA unit tests (Core GEMM/Quantization only)
if get_option('enable-cuda')
  cuda_test_target = [
    ['unittest_quantize_cuda', ['../unittest_util.cpp']],
    ['unittest_int4_utils_cuda', []],
    ['unittest_gemm_fp32_qint4_comparison', ['../unittest_util.cpp']],
    ['unittest_rope_reference', ['../unittest_util.cpp']],
  ]

  foreach target : cuda_test_target
    exe = executable(
      target[0],
      [target[0] + '.cpp'] + target[1],
      include_directories: [
        unittest_inc, 
        include_directories('../../../nntrainer/cuda'), 
        include_directories('../../../nntrainer/utils'),
        include_directories('../../../nntrainer/tensor'),
        include_directories('../../../nntrainer'),
      ],
      dependencies: [unittest_nntrainer_deps, cuda_dep],
      install: get_option('enable-test'),
      install_dir: application_install_dir,
      link_args: ['-NODEFAULTLIB:LIBCMT', '-NOIMPLIB', '-NOEXP']
    )
    test(target[0], exe, args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), target[0])])
  endforeach
endif
