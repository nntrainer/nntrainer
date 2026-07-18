res_path = meson.current_source_dir() / '..' / 'res'
nntr_reinforcement_resdir = nntr_app_resdir / 'ReinforcementLearning'
run_command(['mkdir', '-p', nntr_reinforcement_resdir], check: true)
nntr_deepq_resdir = nntr_reinforcement_resdir / 'DeepQ'
run_command(['cp', '-lr', res_path, nntr_deepq_resdir], check: true)

env_dir='../../Environment'

# DeepQ is the only Application that pulls in jsoncpp + libcurl. Make
# those optional so a host without the dev packages can still build the
# rest of the tree (libnntrainer, CausalLM, nntr_quantize). If either
# dep is missing we skip the DeepQ executable build entirely; the
# remaining Applications are unaffected.
jsoncpp_dep = dependency('jsoncpp', required: false)
libcurl_dep = dependency('libcurl', required: false)

if not jsoncpp_dep.found() or not libcurl_dep.found()
  warning('Applications/ReinforcementLearning/DeepQ: jsoncpp or libcurl ' +
          'not found, skipping DeepQ build. Install libjsoncpp-dev / ' +
          'libcurl4-openssl-dev to enable.')
  subdir_done()
endif

deepq_sources = [
  'main.cpp',
  env_dir / 'CartPole/cartpole.cpp'
]

deepq_inc = [
  env_dir
]

deepq_deps=[
  iniparser_dep,
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
