fs = import('fs')

if get_option('platform') == 'android'
  # nntrainer standalone app build (S1): only CausalLM is wired for a
  # meson-driven android build today. The legacy sample apps below
  # (LogisticRegression, MNIST, VGG, ...) build via ndk-build/Android.mk
  # instead and haven't been validated under a meson android cross build,
  # so they stay out of this path (the caller in the parent meson.build only
  # reaches this subdir on android when enable-transformer is also set).
  if get_option('enable-transformer')
    subdir('CausalLM')
  endif
else
  nntr_app_resdir = nntrainer_resdir / 'app'

  if build_machine.system() == 'windows'
    nntr_app_resdir_win = nntr_app_resdir.replace('/', '\\')
    if not fs.exists (nntr_app_resdir_win)
      run_command([prog_win_cmd, '/C', 'mkdir', nntr_app_resdir_win], check: true)
    endif
  else
    run_command(['mkdir', '-p', nntr_app_resdir], check: true)
  endif

  subdir('utils')
  subdir('LogisticRegression/jni')
  if enable_ccapi
    subdir('MNIST/jni')
    if get_option('enable-fp16')
      subdir('MixedPrecision/jni')
    endif
  endif
  subdir('VGG/jni')
  subdir('Resnet/jni')
  subdir('Multi_input/jni')

  # if enable_capi
  #   subdir('TransferLearning/Draw_Classification/jni')
  # endif

  subdir('ProductRatings/jni')
  subdir('AlexNet/jni')
  subdir('Layers/jni')
  if get_option('enable-tflite-backbone')
    subdir('SimpleShot')
  endif
  subdir('ONNX/jni')
  subdir('SimpleFC/jni')

  if host_machine.system() != 'windows'
    if get_option('enable-tflite-backbone')
      subdir('KNN/jni')
      subdir('TransferLearning/CIFAR_Classification/jni')
    endif
    subdir('YOLOv2/jni')
    subdir('YOLOv3/jni')
    subdir('ReinforcementLearning/DeepQ/jni')
    subdir('Custom')
    if get_option('enable-tflite-interpreter')
    subdir('TFlite_export')
    endif
  endif

  if get_option('enable-transformer')
    if get_option('platform') != 'tizen'
      subdir('CausalLM')
    endif
  endif

  # Neuron backend smoke test. Deliberately gated on enable-neuron alone
  # rather than folded into CausalLM/meson.build: the test only needs the
  # ccapi and the neuron_graph layer, so coupling it to enable-transformer
  # would silently skip it for anyone building just the backend. Desktop-only
  # here — the on-device Android artifact is built via jni/Android.mk.in's
  # nntrainer_neuron_smoke NDK module instead (see NEURON_BUILD.md), since
  # this file's android branch above is narrowed to CausalLM only.
  if get_option('enable-neuron') and enable_ccapi
    subdir('CausalLM/neuron')
  endif

  subdir('Optimizers/jni')
endif
