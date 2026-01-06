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
  subdir('LLaMA/jni')
  if host_machine.system() != 'windows'
  subdir('PicoGPT/jni')
  endif
  if (get_option('platform') != 'tizen') and (get_option('platform') != 'windows')
    subdir('CausalLM')
  endif
endif

subdir('Optimizers/jni')
subdir('XLMRoberta')
