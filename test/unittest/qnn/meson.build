# Host test of nntrainer/qnn/jni/qnn_context_var.h that needs no QNN SDK.
# The header is copied next to the test so that its quoted includes resolve to
# the stub SDK headers in ./stub instead of the real ones in nntrainer/qnn/jni.
configure_file(
  input: meson.source_root() / 'nntrainer' / 'qnn' / 'jni' / 'qnn_context_var.h',
  output: 'qnn_context_var.h',
  copy: true
)

# malloc/calloc/free are wrapped so the test can see which heap blocks
# makeContext() leaves behind.
unittest_qnn_context_var_link_args = [
  '-Wl,--wrap=malloc',
  '-Wl,--wrap=calloc',
  '-Wl,--wrap=free'
]

if cxx.has_multi_link_arguments(unittest_qnn_context_var_link_args)
  unittest_qnn_context_var_exe = executable(
    'unittest_qnn_context_var',
    'unittest_qnn_context_var.cpp',
    include_directories: include_directories('stub'),
    cpp_args: ['-fno-builtin-malloc', '-fno-builtin-calloc', '-fno-builtin-free'],
    link_args: unittest_qnn_context_var_link_args,
    dependencies: [gtest_main_dep],
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test('unittest_qnn_context_var', unittest_qnn_context_var_exe,
    args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_qnn_context_var')],
    timeout: test_timeout,
    suite: 'unittests'
  )
endif
