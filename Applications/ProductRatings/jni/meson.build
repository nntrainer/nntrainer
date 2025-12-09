fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'ProductRatings'

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

e = executable('nntrainer_product_ratings',
  'main.cpp',
  dependencies: [nntrainer_ccapi_dep, gtest_dep, nntrainer_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# test split example
test('app_product_ratings', e, args: ['train',
  build_app_res_dir / 'product_ratings_model.ini',
  build_app_res_dir / 'sample_product_ratings.txt']
)
