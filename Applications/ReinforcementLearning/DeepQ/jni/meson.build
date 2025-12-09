res_path = meson.current_source_dir() / '..' / 'res'
nntr_reinforcement_resdir = nntr_app_resdir / 'ReinforcementLearning'
run_command(['mkdir', '-p', nntr_reinforcement_resdir], check: true)
nntr_deepq_resdir = nntr_reinforcement_resdir / 'DeepQ'
run_command(['cp', '-lr', res_path, nntr_deepq_resdir], check: true)

env_dir='../../Environment'

jsoncpp_dep = dependency('jsoncpp')
libcurl_dep = dependency('libcurl')

deepq_sources = [
  'main.cpp',
  env_dir / 'CartPole/cartpole.cpp'
]

deepq_inc = [
  env_dir
]

deepq_deps=[
  nntrainer_dep,
  tflite_dep,
  jsoncpp_dep,
  libcurl_dep
]

if get_option('use_gym')
   message('[Applications/ReinforcementLearning] Eabling OpenAI/Gym using gym-http-api')
   deepq_deps += dependency('gym-http-api')
   deepq_deps += dependency('boost')
endif

e = executable('nntrainer_deepq',
  deepq_sources,
  dependencies: deepq_deps,
  include_directories: include_directories(env_dir),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# test('app_DeepQ', e, args: [res_path / 'DeepQ.ini'], timeout: 60)
