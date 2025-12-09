fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'AlexNet'

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

alex_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

executable('nntrainer_alexnet',
  alex_sources,
  dependencies: [ nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.'), cifar_include_dir],
  install: get_option('install-app'),
  install_dir: application_install_dir
)
