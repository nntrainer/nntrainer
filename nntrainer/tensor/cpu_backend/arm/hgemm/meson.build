hgemm_headers = [
  'hgemm.h',
  'hgemm_util.h',
  'hgemm_pack.h',
  'hgemm_common.h',
  'hgemm_noTrans.h',
  'hgemm_transA.h',
  'hgemm_transAB.h',
  'hgemm_transB.h'
]

subdir('hgemm_kernel')
nntrainer_inc += include_directories('hgemm_kernel')
nntrainer_inc_abs += meson.current_source_dir() / 'hgemm_kernel'

subdir('hgemm_padding')
nntrainer_inc += include_directories('hgemm_padding')
nntrainer_inc_abs += meson.current_source_dir() / 'hgemm_padding'

hgemm_sources = [
    'hgemm.cpp',
    'hgemm_util.cpp',
    'hgemm_pack.cpp',
    'hgemm_noTrans.cpp',
    'hgemm_transA.cpp',
    'hgemm_transB.cpp',
    'hgemm_transAB.cpp',
]

foreach s : hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

