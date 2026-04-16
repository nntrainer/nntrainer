# KleidiAI qsi4cxp matmul kernels
# NEON dotprod/i8mm: local versions (pre-upstream C99 restrict migration)
# SME/SME2: from upstream ARM-software/kleidiai (2026-04), require C99

matmul_clamp_f32_qai8dxp_qsi4cxp_headers = [
    'kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h',
    # NEON dotprod
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h',
    # NEON i8mm
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h',
]

matmul_clamp_f32_qai8dxp_qsi4cxp_sources = [
    # NEON dotprod
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.c',
    # NEON i8mm
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.c',
]

# SME/SME2 kernels from upstream KleidiAI. These use C99 restrict
# and mixed declarations, so they need GCC 14+ or Clang 16+ (NDK r27).
# GCC 13 (Ubuntu 24.04 cross-compiler) does NOT support them.
if get_option('enable-sme')
  matmul_clamp_f32_qai8dxp_qsi4cxp_headers += [
    'kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot.h',
  ]
  matmul_clamp_f32_qai8dxp_qsi4cxp_sources += [
    'kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.c',
    'kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa_asm.S',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot.c',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot_asm.S',
  ]
endif

if get_option('enable-sme2')
  matmul_clamp_f32_qai8dxp_qsi4cxp_headers += [
    'kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h',
  ]
  matmul_clamp_f32_qai8dxp_qsi4cxp_sources += [
    'kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.c',
    'kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.c',
  ]
endif


foreach s : matmul_clamp_f32_qai8dxp_qsi4cxp_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qai8dxp_qsi4cxp_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
