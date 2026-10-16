tensor_sources = [
  'cache_elem.cpp',
  'cache_loader.cpp',
  'cache_pool.cpp',
  'lazy_tensor.cpp',
  'manager.cpp',
  'tensor.cpp',
  'tensor_base.cpp',
  'float_tensor.cpp',
  'int4_utils.cpp',
  'uint4_tensor.cpp',
  'uint_tensor.cpp',
  'char_tensor.cpp',
  'short_tensor.cpp',
  'q4_k_tensor.cpp',
  'q6_k_tensor.cpp',
  'q4_0_tensor.cpp',
  'q4_0_utils.cpp',
  'htp_q4_0_convert.cpp',
  'htp_act_quant.cpp',
  'qs4cx_tensor.cpp',
  'ternary_tensor.cpp',
  'qs4acx_tensor.cpp',
  'tensor_dim.cpp',
  'var_grad.cpp',
  'weight.cpp',
  'quantizer.cpp',
  'basic_planner.cpp',
  'memory_pool.cpp',
  'swap_device.cpp',
  'tensor_pool.cpp',
  'optimized_v1_planner.cpp',
  'optimized_v2_planner.cpp',
  'optimized_v3_planner.cpp',
  'task_executor.cpp',
]

tensor_headers = [
  'memory_data.h',
  'residency_policy.h',
  'tensor.h',
  'tensor_base.h',
  'float_tensor.h',
  'int4_utils.h',
  'uint4_tensor.h',
  'char_tensor.h',
  'short_tensor.h',
  'q6_k_tensor.h',
  'uint_tensor.h',
  'q4_k_tensor.h',
  'q4_0_tensor.h',
  'q4_0_utils.h',
  'htp_q4_0_convert.h',
  'htp_wh_layout.h',
  'htp_wh_palette.h',
  'htp_fixed_opts.h',
  'swiglu_det.h',
  'htp_act_quant.h',
  'qs4cx_tensor.h',
  'ternary_tensor.h',
  'tq2cx_layout.h',
  'qs4acx_tensor.h',
  'moe_route_det.h',
  'weight.h',
  'var_grad.h',    
  'quantizer.h',    
  'tensor_wrap_specs.h',
  'manager.h',
  'basic_planner.h',
  'memory_planner.h',
  'tensor_pool.h',
  'cache_loader.h',
  'task_executor.h',
  'cache_pool.h',
  'cache_elem.h',
  'memory_pool.h',
  'swap_device.h',
  'task.h'
]

# htp_wh_palette.h includes "htp_backend/hvx/hvx_expand_i2i4.h" (stdint only).
# Installed headers are flat, so install it under htp_backend/hvx/ next to them
# to make that relative include resolve for out-of-tree consumers.
install_headers('htp_backend/hvx/hvx_expand_i2i4.h',
  install_dir: nntrainer_includedir / 'htp_backend' / 'hvx')

subdir('cpu_backend')
nntrainer_inc += include_directories('cpu_backend')
nntrainer_inc_abs += meson.current_source_dir() / 'cpu_backend'

if get_option('enable-fp16')
  tensor_headers += 'half_tensor.h'
  tensor_sources += 'half_tensor.cpp'
endif

if get_option('enable-biqgemm')
  tensor_headers += 'bcq_tensor.h'
  tensor_sources += 'bcq_tensor.cpp'
  nntrainer_inc += biqgemm_inc
  fs = import('fs')
  if fs.is_absolute(biqgemm_path)
    nntrainer_inc_abs += biqgemm_path
  else
    nntrainer_inc_abs += meson.source_root() / biqgemm_path
  endif
endif

if get_option('enable-opencl')
  subdir('cl_operations')
  nntrainer_inc += include_directories('cl_operations')
  nntrainer_inc_abs += meson.current_source_dir() / 'cl_operations'
endif

if get_option('enable-htp')
  subdir('htp_backend')
  nntrainer_inc += include_directories('htp_backend')
  nntrainer_inc_abs += meson.current_source_dir() / 'htp_backend'
  # Hexagon SDK system headers (<remote.h>, <AEEStdErr.h>) that the Stage 2
  # FastRPC client stub (docs/htp_attention/40_moe_ffn_htp_task.md) will need.
  # hexagon_sdk_root is set in the top-level meson.build HTP block
  # (fs.parent(fs.parent(hexkl_root))) and is in meson's flat scope here.
  nntrainer_inc_abs += hexagon_sdk_root / 'incs'
  nntrainer_inc_abs += hexagon_sdk_root / 'incs' / 'stddef'
endif

foreach s : tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
