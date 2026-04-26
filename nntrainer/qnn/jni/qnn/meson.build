nntrainer_inc += include_directories('.')
nntrainer_inc += include_directories('./QNN', './Log', './PAL/include', './Utils', './WrapperUtils')
nntrainer_inc_abs += meson.current_source_dir() / 'QNN'
nntrainer_inc_abs += meson.current_source_dir()

qnn_elem = [
  'Log',
  'op',
  'PAL',
  'Utils',
  'WrapperUtils'
]

qnn_headers = []

qnn_headers_ =[
  'QNN.hpp',
  'QnnTypeMacros.hpp',
  'qnn_properties.h'
]

nntrainer_sources += meson.current_source_dir() / 'qnn_properties.cpp'

foreach h : qnn_headers_
  nntrainer_headers += meson.current_source_dir() / h
endforeach

foreach e : qnn_elem
  subdir(e)
  nntrainer_inc += include_directories(e)
  nntrainer_inc_abs += meson.current_source_dir() / e
endforeach

  nntrainer_inc_abs += meson.current_source_dir() / 'PAL' / 'include'

