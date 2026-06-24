nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

# meson provides include_directories. But there is no support to get the absolute path from the object.
# to feed to the android.mk, sadly nntrainer_inc_abs is saved as well
nntrainer_inc_abs = [
  meson.current_source_dir(),
  meson.source_root() / 'api',
  meson.source_root() / 'api' / 'ccapi' / 'include'
]

nntrainer_sources = []
nntrainer_headers = [
   meson.current_source_dir() / 'nntrainer_log.h',
   meson.current_source_dir() / 'nntrainer_logger.h',
   meson.current_source_dir() / 'nntrainer_error.h',
   meson.current_source_dir() / 'app_context.h',
   meson.current_source_dir() / 'context.h',
   meson.current_source_dir() / 'context_data.h',
   meson.current_source_dir() / 'engine.h',
   meson.current_source_dir() / 'mem_allocator.h',
]

nntrainer_base_deps=[
  blas_dep,
  iniparser_dep,
  libdl_dep,
  libm_dep,
  ml_api_common_dep,
  ruy_dep,
  thread_dep,
]

if get_option('enable-opencl')
  nntrainer_base_deps += clblast_dep
endif

if get_option('platform') == 'tizen'
  nntrainer_base_deps += dependency('dlog')
endif

nntrainer_elements = [
  'compiler',
  'schema',
  'dataset',
  'layers',
  'models',
  'optimizers',
  'tensor',
  'utils',
  'utils/ddtree',
  'graph'
]

if get_option('enable-opencl')
  nntrainer_elements += 'opencl'
  nntrainer_elements += 'layers/cl_layers'

  nntrainer_inc_abs += meson.source_root() / 'subprojects' / 'CLBlast' / 'include'
endif

if get_option('enable-npu')
  nntrainer_elements += 'qnn'
  # Note: qnn subdir is NOT processed here with the other elements.
  # libqnn_context.so depends on nntrainer_dep, which is declared later.
  # The qnn subdir is processed after nntrainer_dep is available (see below).
endif

foreach elem : nntrainer_elements
  if elem != 'qnn'
    subdir(elem)
  endif
  nntrainer_inc += include_directories(elem)
  nntrainer_inc_abs += meson.current_source_dir() / elem
endforeach

nntrainer_common_sources = [
  'nntrainer_logger.cpp',
  'app_context.cpp',
  'engine.cpp',
  'mem_allocator.cpp'
]

# QNN context is built as a separate plugin .so (libqnn_context.so)
# and loaded dynamically by Engine. See qnn/jni/meson.build for the
# shared_library target. The header is still needed for type definitions.
if get_option('enable-npu')
  nntrainer_headers += meson.current_source_dir() / 'qnn' / 'qnn_context.h'
endif

nntrainer_inc_abs += meson.source_root() / 'subprojects' / 'ruy'

if get_option('enable-opencl')
  nntrainer_headers += meson.current_source_dir() / 'cl_context.h'
  nntrainer_headers += meson.current_source_dir() / 'cl_svm_allocator.h'
  nntrainer_common_sources += 'cl_context.cpp'
  nntrainer_common_sources += 'cl_svm_allocator.cpp'
endif

foreach s : nntrainer_common_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

if get_option('platform') == 'android'
  nntrainer_dep = declare_dependency(include_directories: nntrainer_inc)
else
  # Build libraries

  nntrainer_static = static_library('nntrainer',
    nntrainer_sources,
    dependencies: nntrainer_base_deps,
    include_directories: nntrainer_inc,
    install: true,
    install_dir: nntrainer_libdir
  )

  if get_option('platform') == 'windows'
    nntrainer_def = custom_target('nntrainer_def',
    command : ['python', meson.source_root() / 'tools' / 'pyutils' / 'generate_def.py', '--objects_dir', meson.build_root() / 'nntrainer' / 'libnntrainer.a.p', '--working_dir', meson.current_build_dir()],
    output: 'nntrainer.def',
    depends: nntrainer_static,
    )

    nntrainer_shared = shared_library('nntrainer',
      nntrainer_sources,
      dependencies: nntrainer_base_deps,
      include_directories: nntrainer_inc,
      install: true,
      install_dir: nntrainer_libdir,
      vs_module_defs : nntrainer_def
    )
  else
    nntrainer_shared = shared_library('nntrainer',
      nntrainer_sources,
      dependencies: nntrainer_base_deps,
      include_directories: nntrainer_inc,
      install: true,
      install_dir: nntrainer_libdir
    )
  endif

  nntrainer_lib = nntrainer_shared
  if get_option('default_library') == 'static'
    nntrainer_lib = nntrainer_static
  endif

  nntrainer_dep = declare_dependency(link_with: nntrainer_lib,
    dependencies: nntrainer_base_deps,
    include_directories: nntrainer_inc
  )
endif

# Build QNN context plugin after nntrainer_dep is available.
# libqnn_context.so links against libnntrainer, so it must be built after it.
if get_option('enable-npu')
  subdir('qnn')
endif

# Install headers
install_headers(nntrainer_headers,
  install_dir: nntrainer_includedir,
)
