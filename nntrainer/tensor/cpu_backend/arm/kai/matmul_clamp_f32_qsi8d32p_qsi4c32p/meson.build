matmul_clamp_f32_qsi8d32p_qsi4c32p_headers = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h',
    'kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h'
]

matmul_clamp_f32_qsi8d32p_qsi4c32p_sources = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c',
]

# SVE kernels: use enable-sve option instead of cxx.has_argument, because
# for Android ndk-build the meson host compiler is x86 and can't test
# ARM -march flags. The actual compilation happens via NDK Clang which
# DOES support SVE.
if get_option('enable-sve') or get_option('enable-sve2')
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
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S',
  ]
endif

foreach s : matmul_clamp_f32_qsi8d32p_qsi4c32p_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qsi8d32p_qsi4c32p_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
