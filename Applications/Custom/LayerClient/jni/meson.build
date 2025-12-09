fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'LayerClient'

if build_machine.system() == 'windows'
  app_res_dir_win = app_res_dir.replace('/', '\\')
  build_app_res_dir_win = build_app_res_dir.replace('/', '\\')
  if not fs.exists (build_app_res_dir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', build_app_res_dir_win], check: true)
  endif
  run_command(['xcopy', '/C', '/Y', app_res_dir_win, build_app_res_dir_win], check: true)
else
  run_command(['cp', '-lr', app_res_dir, build_app_res_dir], check: true)
endif

layer_client_sources = [
  'main.cpp',
  layer_example_pow_src,
  layer_example_mae_src
]

exe = executable('layer_client',
  layer_client_sources,
  include_directories: layer_example_inc,
  dependencies: [app_utils_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_layer_client_api', exe, args: 'model')
test('app_layer_client_ini', exe, args: build_app_res_dir / 'custom_layer_client.ini')
