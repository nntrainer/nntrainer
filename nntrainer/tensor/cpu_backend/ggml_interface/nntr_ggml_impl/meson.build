ggml_impl_interface_headers = [
    'nntr_ggml_impl.h',
    'nntr_ggml_impl_common.h',
    'nntr_ggml_impl_utils.h',
]

# Common sources always compiled
ggml_impl_interface_sources = [
  'nntr_ggml_impl_quant.cpp',
  'nntr_ggml_impl_fp16_fp32.cpp',
]

# Build-time exclusive source selection (ARM first, then x86)
arch = host_machine.cpu_family()

# Since android build uses ndk-build, meson can't detect its arch properly. (arch != aarch64)
# So we need to add an android case explicitly
if arch == 'aarch64' or get_option('platform') == 'android'
  ggml_impl_interface_sources += 'nntr_ggml_impl_neon.cpp'
elif arch == 'arm'
  # @todo check neon_impl for armv7
  ggml_impl_interface_sources += 'nntr_ggml_impl_fallback.cpp'
elif arch == 'x86_64' or arch == 'x86'
  # AVX/AVX2/AVX512 implementation for x86_64
  ggml_impl_interface_sources += 'nntr_ggml_impl_avx.cpp'
  if enable_avxvnni_dispatch
    # Second build of the Q4_0 8x8 int8 kernels with -mavxvnni, chosen at run
    # time by cpuid (see nntr_ggml_impl_avx.cpp). Its own target, so the flag
    # never reaches code that runs on AVX2-only CPUs.
    nntrainer_link_whole += static_library('nntr_ggml_impl_avxvnni',
      'nntr_ggml_impl_avxvnni.cpp',
      cpp_args: ['-mavxvnni'],
      dependencies: nntrainer_base_deps,
      include_directories: [nntrainer_inc, include_directories('.', '../..', '../../..')],
      pic: true,
    )
  endif
else
  # Fallback for other architectures
  ggml_impl_interface_sources += 'nntr_ggml_impl_fallback.cpp'
endif

foreach s : ggml_impl_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : ggml_impl_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

