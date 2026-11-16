kai_headers = [
    'kai_common.h'
]

if get_option('enable-fp16')
subdir('matmul_clamp_f16_qai8dxp_qsi4cxp')
nntrainer_inc += include_directories('matmul_clamp_f16_qai8dxp_qsi4cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f16_qai8dxp_qsi4cxp'
endif

subdir('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc += include_directories('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qai8dxp_qsi4cxp'

subdir('matmul_clamp_f32_qai8dxp_qsi8cxp')
nntrainer_inc += include_directories('matmul_clamp_f32_qai8dxp_qsi8cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qai8dxp_qsi8cxp'

subdir('pack')
nntrainer_inc += include_directories('pack')
nntrainer_inc_abs += meson.current_source_dir() / 'pack'

foreach h : kai_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
