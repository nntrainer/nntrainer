# SPDX-License-Identifier: Apache-2.0

pack_sources = [
    'kai_lhs_quant_pack_qsi8d32p_f32.c',
    'kai_lhs_quant_pack_qsi8d32p_f32_neon.c',
    'kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c',
]

pack_headers = [
    'kai_lhs_quant_pack_qsi8d32p_f32.h',
    'kai_lhs_quant_pack_qsi8d32p_f32_neon.h',
    'kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h',
]

foreach s : pack_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : pack_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
