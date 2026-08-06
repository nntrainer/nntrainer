if get_option('default_library') != 'static' and not meson.is_cross_build()
  singleton_probe_cpp_args = []
  if host_machine.system() != 'windows'
    singleton_probe_cpp_args += ['-fvisibility=hidden']
  endif

  singleton_probe_a = shared_library(
    'nntrainer_singleton_probe_a',
    'singleton_probe_a.cpp',
    cpp_args: singleton_probe_cpp_args + ['-DNNTR_SINGLETON_PROBE_A_EXPORTS'],
    dependencies: nntrainer_dep,
    install: false
  )

  singleton_probe_b = shared_library(
    'nntrainer_singleton_probe_b',
    'singleton_probe_b.cpp',
    cpp_args: singleton_probe_cpp_args + ['-DNNTR_SINGLETON_PROBE_B_EXPORTS'],
    dependencies: nntrainer_dep,
    install: false
  )

  singleton_dso_test = executable(
    'unittest_singleton_dso',
    'unittest_singleton_dso.cpp',
    link_with: [singleton_probe_a, singleton_probe_b],
    dependencies: [nntrainer_dep, thread_dep],
    install: false
  )

  singleton_dso_test_env = environment()
  singleton_dso_test_env.set('NNTR_NUM_THREADS', '1')
  test(
    'unittest_singleton_dso',
    singleton_dso_test,
    env: singleton_dso_test_env,
    timeout: test_timeout,
    suite: 'unittests'
  )
endif
