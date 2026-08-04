# nntrainer/nntrainer/neuron/meson.build
# Defines neuron_sources_abs and neuron_inc_abs for the ndk-build Android.mk
# module. Gated by the parent (enable-neuron).
#
# Unlike nntrainer/qnn/meson.build, there is no configure-time vendor-tree
# generation step here: RuntimeAPI.h and Types.h are two small,
# self-contained headers that are checked into vendor/ directly (see
# vendor/README.md) rather than pulled from a locally-installed SDK at
# configure time.

neuron_src_dir = meson.current_source_dir()

neuron_sources_abs = [
  neuron_src_dir / 'neuron_api.cpp',
  neuron_src_dir / 'neuron_context.cpp',
  neuron_src_dir / 'neuron_dma_allocator.cpp',
  neuron_src_dir / 'neuron_properties.cpp',
  neuron_src_dir / 'NeuronGraph.cpp',
]

neuron_inc_abs = [
  neuron_src_dir,
  neuron_src_dir / 'vendor',
]

neuron_inc = include_directories(
  '.',
  'vendor',
)

# Expose neuron headers (neuron_context.h etc.) to the rest of nntrainer.
nntrainer_inc += neuron_inc
nntrainer_inc_abs += neuron_src_dir
