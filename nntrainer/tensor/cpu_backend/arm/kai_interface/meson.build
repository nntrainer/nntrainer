kai_interface_headers = [
    'kleidiai_interface.h',
]
kai_interface_sources = [
    'kleidiai_interface_qai8dxp_qsi4cxp.cpp',
    'kleidiai_interface_qai8dxp_qsi8cxp.cpp',
]

if get_option('enable-fp16')
kai_interface_sources += [
  'kleidiai_interface_f16_qai8dxp_qsi4cxp.cpp',
]
endif

subdir('kai')
nntrainer_inc += include_directories('kai')
nntrainer_inc_abs += meson.current_source_dir() / 'kai'

foreach s : kai_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : kai_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
