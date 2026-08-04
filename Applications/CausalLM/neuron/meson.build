# nntrainer/Applications/CausalLM/neuron/meson.build
# Smoke-test executable for the Neuron backend.

neuron_app_sources = [
  'nntrainer_neuron_smoke.cpp',
]

neuron_smoke_exe = executable(
  'nntrainer_neuron_smoke',
  neuron_app_sources,
  dependencies: [ccapi_dep, nntrainer_dep],
  include_directories: [nntrainer_inc],
  install: true,
  install_dir: nntrainer_bindir,
)
