# The x86 hgemm headers are internal (nntrainer::hgemm::internal) and are used
# only through the cpu_backend dispatch inside this library, so they are added to
# the include path below but intentionally not installed.

subdir('hgemm_kernel')
nntrainer_inc += include_directories('hgemm_kernel')
nntrainer_inc_abs += meson.current_source_dir() / 'hgemm_kernel'

x86_hgemm_sources = [
  'hgemm.cpp',
  'hgemm_blocked.cpp',
  'hgemm_fast_path.cpp',
  'hgemm_pack.cpp',
  'hgemm_util.cpp',
  'hgemm_workspace.cpp',
]

foreach s : x86_hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
