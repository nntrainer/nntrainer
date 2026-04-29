cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

transpose_src = files('transpose_layer.cpp')
transpose_layer = static_library('transpose',
  transpose_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: cpp_args
)
transpose_dep = declare_dependency(
  link_with: transpose_layer,
  include_directories: include_directories('./')
)

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
  run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.source_root(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
  run_command([meson.source_root() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

rms_norm_src = files('rms_norm.cpp')
rms_norm_layer = static_library('rms_norm',
  rms_norm_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: cpp_args
)
rms_norm_dep = declare_dependency(
  link_with: rms_norm_layer,
  include_directories: include_directories('./')
)


rotary_emb_src = files('rotary_embedding.cpp')
rotary_emb_layer = static_library('rotary_embedding',
  rotary_emb_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: cpp_args
)
rotary_emb_dep = declare_dependency(
  link_with: rotary_emb_layer,
  include_directories: include_directories('./')
)

mha_src = files('custom_multi_head_attention_layer.cpp')
mha_layer = static_library('custom_multi_head_attention_layer',
  mha_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: cpp_args
)
mha_dep = declare_dependency(
  link_with: mha_layer,
  include_directories: include_directories('./')
)

llama_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp',
  rms_norm_src,
  rotary_emb_src,
  mha_src
]

llama_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  transpose_dep,
  rms_norm_dep,
  rotary_emb_dep,
  mha_dep
]

e = executable('nntrainer_llama',
  llama_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: llama_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
