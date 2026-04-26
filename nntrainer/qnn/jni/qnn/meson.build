# nntrainer-authored QNN-side files. The Qualcomm SDK headers and sample
# utility tree (Log/, PAL/, Utils/, WrapperUtils/, QNN/) used to live as
# siblings of this directory; they were removed when the SDK was made
# external (see ../../meson.build). The build now relies on -Dqnn-sdk-root
# for those headers.

nntrainer_inc += include_directories('.')
nntrainer_inc_abs += meson.current_source_dir()

nntrainer_sources += meson.current_source_dir() / 'qnn_properties.cpp'

qnn_headers = [
  'qnn_properties.h',
]

foreach h : qnn_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

subdir('op')
nntrainer_inc += include_directories('op')
nntrainer_inc_abs += meson.current_source_dir() / 'op'
