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
arm_arch_opt = get_option('arm-arch')

if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  if arm_arch_opt == 'armv9.2-a'
    # SVE/SVE2 implementation for armv9.2-a
    ggml_impl_interface_sources += 'nntr_ggml_impl_sve.cpp'
  else
    # NEON implementation (armv8.2-a, armv7l, or auto)
    ggml_impl_interface_sources += 'nntr_ggml_impl_neon.cpp'
  endif
elif arch == 'x86_64' or arch == 'x86'
  # AVX/AVX2/AVX512 implementation for x86_64
  ggml_impl_interface_sources += 'nntr_ggml_impl_avx.cpp'
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

