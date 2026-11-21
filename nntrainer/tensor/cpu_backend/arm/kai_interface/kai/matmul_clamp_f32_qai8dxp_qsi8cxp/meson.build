matmul_clamp_f32_qai8dxp_qsi8cxp_headers = [
    'kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h',
]

matmul_clamp_f32_qai8dxp_qsi8cxp_sources = [
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c',
]

# Issues i8mm instructions; on aarch64 it compiles without
# __ARM_FEATURE_MATMUL_INT8, so it has to be left out explicitly. The
# interface table references it under the same condition.
if get_option('enable-i8mm')
matmul_clamp_f32_qai8dxp_qsi8cxp_headers += [
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h',
]
matmul_clamp_f32_qai8dxp_qsi8cxp_sources += [
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c',
]
endif

foreach s : matmul_clamp_f32_qai8dxp_qsi8cxp_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qai8dxp_qsi8cxp_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
