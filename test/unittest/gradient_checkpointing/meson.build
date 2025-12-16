test_name = 'unittest_gradient_checkpointing'

test_target = [
  'gradient_checkpointing_test_util.cpp',
  'unittest_gradient_checkpointing.cpp'
]

exe = executable(
  test_name,
  test_target,
  dependencies: [
    nntrainer_test_main_deps,
    nntrainer_dep,
    nntrainer_ccapi_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)


test(test_name, exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name),
  timeout: test_timeout,
  suite: 'unittests'
)
