compiler_sources = [
  'ini_interpreter.cpp',
  'activation_realizer.cpp',
  'fusion_realizer.cpp',
  'flatten_realizer.cpp',
  'recurrent_realizer.cpp',
  'remap_realizer.cpp',
  'slice_realizer.cpp',
  'input_realizer.cpp',
  'previous_input_realizer.cpp',
  'multiout_realizer.cpp',
  'bn_realizer.cpp',
  'tflite_export_realizer.cpp',
]

compiler_headers = [
  'compiler_fwd.h'
]

if get_option('enable-onnx-interpreter')
  compiler_sources += [
    'onnx_interpreter.cpp',
  ]
  compiler_headers += [
    'onnx_interpreter.h',
  ]
endif

if get_option('enable-tflite-interpreter')
  compiler_sources += [
    'tflite_interpreter.cpp',
    'tflite_opnode.cpp'
  ]

if not flatc_prog.found()
  error('flatc executable not found')
endif
endif

foreach s : compiler_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : compiler_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
