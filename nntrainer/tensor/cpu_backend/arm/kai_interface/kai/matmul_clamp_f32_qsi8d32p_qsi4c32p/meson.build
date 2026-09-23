matmul_clamp_f32_qsi8d32p_qsi4c32p_headers = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h'
]

matmul_clamp_f32_qsi8d32p_qsi4c32p_sources = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c',
]

# Needs __ARM_FEATURE_MATMUL_INT8 to compile; see the interface table.
if nntr_arm_i8mm
matmul_clamp_f32_qsi8d32p_qsi4c32p_headers += [
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h',
]
matmul_clamp_f32_qsi8d32p_qsi4c32p_sources += [
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c',
]
endif

# The interface table references these under #ifdef __ARM_FEATURE_SVE, so they
# have to follow whether the resolved ISA actually has SVE.
if nntr_arm_sve
  matmul_clamp_f32_qsi8d32p_qsi4c32p_headers += [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h',
  ]
  matmul_clamp_f32_qsi8d32p_qsi4c32p_sources += [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod_asm.S',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_asm.S',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S'
  ]
endif

foreach s : matmul_clamp_f32_qsi8d32p_qsi4c32p_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qsi8d32p_qsi4c32p_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
