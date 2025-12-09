res_path = meson.current_source_dir() / '..' / 'res'
nntr_pico_gpt_resdir = nntr_app_resdir / 'PicoGPT'
run_command(['cp', '-lr', res_path, nntr_pico_gpt_resdir], check: true)

if get_option('platform') != 'tizen'
    run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.1'], check: true)
endif

pico_gpt_sources = [
  'main.cpp',
]

pico_gpt_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  pico_gpt_dependencies += [gtest_dep]
endif

e = executable('nntrainer_pico_gpt',
  pico_gpt_sources,
  include_directories: [include_directories('.')],
  dependencies: pico_gpt_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
