fallback_headers = ['fallback_internal.h']
fallback_sources = ['fallback_internal.cpp']

arch = host_machine.cpu_family()

if arch != 'arm' and arch != 'aarch64' and arch != 'x86_64' and arch != 'x86'
  fallback_headers += 'fallback.h'
  fallback_sources += 'fallback.cpp'
  if get_option('enable-fp16')
    fallback_sources += 'fallback_fp16.cpp'
  endif
endif

if get_option('enable-fp16')
  fallback_sources += 'fallback_internal_fp16.cpp'
endif

foreach s : fallback_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : fallback_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
