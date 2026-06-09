kai_interface_headers = [
    'kleidiai_interface.h',
]
kai_interface_sources = [
    'kleidiai_interface_qai8dxp_qsi4cxp.cpp',
    'kleidiai_interface_qsi8d32p_qsi4c32p.cpp',
]

subdir('kai')
nntrainer_inc += include_directories('kai')
nntrainer_inc_abs += meson.current_source_dir() / 'kai'

# Select thread backend for kleidiai parallel implementation
if get_option('thread-backend') == 'bstp'
  kai_interface_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_stdthread.cpp'
elif get_option('thread-backend') == 'omp'
  kai_interface_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
elif get_option('thread-backend') == 'mixed'
  kai_interface_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
else
  message('Unknown thread-backend meson option! Use omp as a default for kleidiai.')
  kai_interface_sources += 'kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp'
endif


foreach s : kai_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : kai_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
