kai_headers = [
    'kai_common.h'
]

# SVE/SME common ASM helpers (runtime vector-length queries, SME commit)
kai_sources = []
if get_option('enable-sve') or get_option('enable-sve2')
  kai_sources += 'kai_common_sve_asm.S'
endif
if get_option('enable-sme') or get_option('enable-sme2')
  kai_sources += 'kai_common_sme_asm.S'
endif

subdir('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc += include_directories('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qai8dxp_qsi4cxp'

subdir('matmul_clamp_f32_qsi8d32p_qsi4c32p')
nntrainer_inc += include_directories('matmul_clamp_f32_qsi8d32p_qsi4c32p')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qsi8d32p_qsi4c32p'

subdir('pack')
nntrainer_inc += include_directories('pack')
nntrainer_inc_abs += meson.current_source_dir() / 'pack'

foreach s : kai_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : kai_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
