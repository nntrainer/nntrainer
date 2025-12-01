optimizer_sources = [
  'adam.cpp',
  'optimizer_devel.cpp',
  'sgd.cpp',
  'optimizer_context.cpp',
  'lr_scheduler_constant.cpp',
  'lr_scheduler_cosine.cpp',
  'lr_scheduler_exponential.cpp',
  'lr_scheduler_linear.cpp',
  'lr_scheduler_step.cpp',
  'optimizer_wrapped.cpp',
  'adamw.cpp',
  'lion.cpp',
]

optimizer_headers = [
  'optimizer_devel.h',
  'optimizer_context.h',
  'lr_scheduler.h',
  'optimizer_wrapped.h'
]

foreach s : optimizer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : optimizer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

