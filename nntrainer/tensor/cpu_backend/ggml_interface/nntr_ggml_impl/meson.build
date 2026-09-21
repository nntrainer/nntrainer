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

message('arm_arch_opt : ' + arm_arch_opt)

# arm_arch_opt != 'none' case for cross-compilation scenario
if arch == 'aarch64' or arm_arch_opt != 'none'
  ggml_impl_interface_sources += 'nntr_ggml_impl_neon.cpp'
elif arch == 'x86_64' or arch == 'x86'
  # AVX/AVX2/AVX512 implementation for x86_64
  ggml_impl_interface_sources += 'nntr_ggml_impl_avx.cpp'
elif arch == 'arm'
  ggml_impl_interface_sources += 'nntr_ggml_impl_fallback.cpp'
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

