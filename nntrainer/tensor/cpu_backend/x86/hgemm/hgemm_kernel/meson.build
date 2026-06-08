x86_hgemm_kernel_headers = [
  'hgemm_kernel.h',
]

x86_hgemm_kernel_sources = [
  'hgemm_kernel_1x8.cpp',
  'hgemm_kernel_1x16.cpp',
  'hgemm_kernel_2x8.cpp',
  'hgemm_kernel_2x16.cpp',
  'hgemm_kernel_4x8.cpp',
  'hgemm_kernel_4x16.cpp',
  'hgemm_kernel_6x8.cpp',
  'hgemm_kernel_6x16.cpp',
  'hgemm_kernel_mxn.cpp',
]

foreach s : x86_hgemm_kernel_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : x86_hgemm_kernel_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
