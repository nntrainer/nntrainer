fs = import('fs')

cl_op_kernels = [
  'addition.cl',
  'concat_axis_1.cl',
  'concat_axis_2.cl',
  'concat_axis_3.cl',
  'convert_block_q4_0.cl',
  'copy.cl',
  'dot.cl',
  'int4_gemv.cl',
  'int4_quantize_input.cl',
  'int4_gemm.cl',
  'int8_int4_gemm_v8c.cl',
  'int8_int8_gemm_xmx.cl',
  'two_conv_attention.cl',
  'flash_attention.cl',
  'q4_0_ab_bi_8x4.cl',
  'q6_k_sgemv.cl',
  'restore_block_q4_0.cl',
  'rmsnorm.cl',
  'layernorm.cl',
  'rotary_emb.cl',
  'sgemm_no_trans.cl',
  'sgemm_trans_a.cl',
  'sgemm_trans_ab.cl',
  'sgemm_trans_b.cl',
  'sgemv_no_trans.cl',
  'sgemv.cl',
  'sscal.cl',
  'swiglu.cl',
  'gelu.cl',
  'geglu.cl',
  'transpose_16bit.cl',
  'transpose_32bit_16bit.cl',
  'transpose_axis_0.cl',
  'transpose_axis_1.cl',
  'transpose_axis_2.cl',
]

if get_option('enable-fp16')
  cl_op_kernels += [
    'addition_fp16.cl',
    'concat_axis_1_fp16.cl',
    'concat_axis_2_fp16.cl',
    'concat_axis_3_fp16.cl',
    'copy_fp16.cl',
    'dot_fp16.cl',
    'hgemm_no_trans.cl',
    'hgemm_trans_a.cl',
    'hgemm_trans_ab.cl',
    'hgemm_trans_b.cl',
    'hgemv_no_trans.cl',
    'hgemv.cl',
    'hscal.cl',
    'rmsnorm_fp16.cl',
    'layernorm_fp16.cl',
    'rotary_emb_fp16.cl',
    'swiglu_fp16.cl',
    'gelu_fp16.cl',
    'geglu_fp16.cl',
    'transpose_axis_0_fp16.cl',
    'transpose_axis_1_fp16.cl',
    'transpose_axis_2_fp16.cl',
  ]
endif

all_includes = ''
foreach k : cl_op_kernels
  basename = fs.stem(k)
  kernel_code = fs.read(k).split('\n')
  modified_code = []
  foreach line : kernel_code
    modified_code += ('R"(' + line + ')"' + ' "\\n"')
  endforeach
  variable_name = basename + '_kernel'
  if basename == 'int4_gemm'
    variable_name = 'gemm_int4_kernel'
  endif
  conf_data = configuration_data(
    {
      'namespace': 'nntrainer',
      'variable_name': variable_name,
      'header_path': basename + '.h',
      'kernel_code': '\n'.join(modified_code),
    },
  )

  configure_file(
    input: 'kernel.cpp.in',
    output: basename + '.cpp',
    configuration: conf_data,
  )
  configure_file(input: 'kernel.h.in', output: basename + '.h', configuration: conf_data)

  # source.full_path() does not work: https://github.com/mesonbuild/meson/issues/5273#issuecomment-1851811417
  # can be used when meson is upgraded to 1.4.0
  # NOTE: the generated per-kernel string headers (e.g. swiglu.h, geglu.h) are
  # INTERNAL to libnntrainer (consumed by the *.cpp below via the build-dir
  # include path). They are intentionally NOT added to nntrainer_headers, i.e.
  # not installed: installing them into <prefix>/include/nntrainer shadows
  # app-side layer headers of the same name (CausalLM's swiglu.h = SwiGLULayer),
  # breaking the app build. Only the generated .cpp goes into nntrainer_sources.
  nntrainer_sources += meson.current_build_dir() / basename + '.cpp'

  all_includes += f'#include "@basename@.h"\n'
endforeach
configure_file(
  input: 'kernels.h.in',
  output: 'cl_kernels.h',
  configuration: {'includes': all_includes},
)

nntrainer_inc += include_directories('..')
nntrainer_inc_abs += meson.current_build_dir() / '..'
