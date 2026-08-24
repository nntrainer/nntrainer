fs = import('fs')

nntr_app_resdir = nntrainer_resdir / 'app'

if build_machine.system() == 'windows'
  nntr_app_resdir_win = nntr_app_resdir.replace('/', '\\')
  if not fs.exists (nntr_app_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntr_app_resdir_win], check: true)
  endif
else
  run_command(['mkdir', '-p', nntr_app_resdir], check: true)
endif

# The android platform build only ever packages CausalLM (see the
# "android app: only building CausalLM (other apps skipped)" warning in the
# top-level meson.build) - the other example apps below are desktop/host
# demos (MNIST, VGG, YOLO, etc.) that were never validated against a host
# compiler + enable-fp16=true combination (several unconditionally declare
# __fp16-typed API surface, which plain host gcc/g++ doesn't support at all -
# unlike NDK clang, which does). Skip them for platform=android so that
# warning's claim is actually true and `ninja install` (which depends on
# building every default target, not just the one this package_android.sh
# run cares about) doesn't fail on unrelated example apps.
if get_option('platform') == 'android'
  if get_option('enable-transformer')
    subdir('CausalLM')
  endif
else
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

  subdir('Optimizers/jni')
endif
