matmul_clamp_f16_qai8dxp_qsi4cxp_headers = [
    'kai_matmul_clamp_f16_qai8dxp_qsi4cxp_interface.h',
    'kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod.h',
    'kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.h',
]

matmul_clamp_f16_qai8dxp_qsi4cxp_sources = [
    'kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod_asm.S',
    'kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_asm.S',
    'kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod.c',
    'kai_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod_asm.S',
    'kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.c',
    'kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_asm.S',
]

foreach s : matmul_clamp_f16_qai8dxp_qsi4cxp_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f16_qai8dxp_qsi4cxp_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
