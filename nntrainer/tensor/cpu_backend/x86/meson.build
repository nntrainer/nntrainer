simd_interface_x86_headers = [
  'x86_compute_backend.h',
  'avx2_impl.h',
  'x86_qsi4cxp.h',
]
simd_interface_x86_sources = [
  'x86_compute_backend.cpp',
  'x86_ops_table.cpp',
  'avx2_impl.cpp',
  'x86_qsi4cxp.cpp',
]

# Note : if we need avx512 support in the future, remove avx2_impl.cpp from the default list
if get_option('enable-fp16')
    simd_interface_x86_sources += 'avx2_impl_fp16.cpp'
    simd_interface_x86_sources += 'x86_compute_backend_fp16.cpp'
endif

foreach s : simd_interface_x86_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_x86_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
