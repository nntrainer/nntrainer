cuda_sources = [
  'cuda_context_manager.cpp',
  'cuda_stream_manager.cpp',
  'cuda_buffer.cpp',
  'cuda_buffer_manager.cpp',
  'cuda_module.cpp',
  'cuda_kernel.cpp',
  'cuda_compute_ops.cpp',
  'cuda_rmsnorm.cpp',
  'cuda_layernorm.cpp',
  'cuda_gelu.cpp',
  'cuda_elementwise.cpp',
  'cuda_blas_manager.cpp',
  'cuda_fc_qs4cx.cpp',
  'cuda_rope.cpp',
  'cuda_attention.cpp',
]


foreach s : cuda_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

# No install_headers() here, deliberately. Every header in this directory is
# an implementation detail of the backend -- device managers, kernel wrappers
# and the op-table internals -- and nothing outside the library includes one.
# A header appended to nntrainer_headers is installed into
# %{_includedir}/nntrainer, and RPM's check-files stage then fails the whole
# Tizen GBS build (and dh_missing --fail-missing the Debian one) for any
# installed file no %files section claims. In-tree includes are unaffected:
# this directory is on nntrainer_inc via nntrainer_elements.
