nntrainer_inc = [
  include_directories('.'),
  include_directories('../api', '../api/ccapi/include')
]

message('SUBDIR: enable-opencl option value: ' + get_option('enable-opencl').to_string())

if get_option('enable-opencl')
  nntrainer_inc += include_directories('opencl')
  message('OpenCL include directory added to nntrainer_inc')
endif

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
   meson.current_source_dir() / 'engine.h',
   meson.current_source_dir() / 'mem_allocator.h',
]

nntrainer_base_deps=[
  blas_dep,
  iniparser_dep,
  libdl_dep,
  libm_dep,
  ml_api_common_dep,
  openmp_dep,
  ruy_dep,
  thread_dep,
]

if get_option('enable-opencl')
  nntrainer_base_deps += clblast_dep
endif

if get_option('enable-cuda')
  # Add CUDA runtime library dependency
  if get_option('platform') == 'windows'
    # Windows: Use CUDA runtime library
    cuda_dep = dependency('cuda', method: 'cmake', required: false)
    if not cuda_dep.found()
      # Fallback to manual library specification for Windows
      cuda_lib = cc.find_library('cudart', required: false)
      if cuda_lib.found()
        nntrainer_base_deps += cuda_lib
      endif
    else
      nntrainer_base_deps += cuda_dep
    endif

    # Also link against cuda.lib for Driver API functions (cuDeviceGet, etc.)
    # Attempt to find CUDA_PATH from environment variable for dynamic path resolution
    cuda_dirs = []
    python_cmd = find_program('python', required: false)
    if python_cmd.found()
      r = run_command(python_cmd, '-c', 'import os; print(os.environ.get("CUDA_PATH", ""))', check: false)
      if r.returncode() == 0
        cuda_env_path = r.stdout().strip()
        if cuda_env_path != ''
          cuda_dirs += [cuda_env_path / 'lib/x64']
        endif
      endif
    endif

    # Find cuda.lib using the dynamically resolved directory or fallback to default paths
    cuda_driver_lib = cc.find_library('cuda', required: false, dirs: cuda_dirs)
    
    if not cuda_driver_lib.found()
        # Fallback: try finding without specific dirs (system path)
        cuda_driver_lib = cc.find_library('cuda', required: false)
    endif
    
    if cuda_driver_lib.found()
        nntrainer_base_deps += cuda_driver_lib
    else
        warning('CUDA driver library (cuda.lib) not found. Driver API calls may fail to link.')
    endif
  else
    # Linux and others: Use pkg-config or manual library specification
    cuda_dep = dependency('cuda', method: 'pkg-config', required: false)
    if not cuda_dep.found()
      # Fallback to manual library specification for Linux
      cuda_lib = cc.find_library('cudart', required: false)
      if cuda_lib.found()
        nntrainer_base_deps += cuda_lib
      endif
    else
      nntrainer_base_deps += cuda_dep
    endif
    
    # On Linux, cuda driver API is usually in libcuda.so
    cuda_driver_lib = cc.find_library('cuda', required: false)
    if cuda_driver_lib.found()
        nntrainer_base_deps += cuda_driver_lib
    endif
  endif
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
  'graph'
]

if get_option('enable-opencl')
  nntrainer_elements += 'opencl'
  nntrainer_elements += 'layers/cl_layers'

  nntrainer_inc_abs += meson.source_root() / 'subprojects' / 'CLBlast' / 'include'
endif

if get_option('enable-cuda')
  nntrainer_elements += 'cuda'
endif

foreach elem : nntrainer_elements
  subdir(elem)
  nntrainer_inc += include_directories(elem)
  nntrainer_inc_abs += meson.current_source_dir() / elem
endforeach

nntrainer_common_sources = [
  'nntrainer_logger.cpp',
  'app_context.cpp',
  'engine.cpp',
  'mem_allocator.cpp'
]

nntrainer_inc_abs += meson.source_root() / 'subprojects' / 'ruy'

if get_option('enable-opencl')
  nntrainer_headers += meson.current_source_dir() / 'cl_context.h'
  nntrainer_common_sources += 'cl_context.cpp'
endif

if get_option('enable-cuda')
  nntrainer_headers += meson.current_source_dir() / 'cuda_context.h'
  nntrainer_headers += meson.current_source_dir() / 'cuda_buffer_manager.h'
  nntrainer_common_sources += 'cuda_context.cpp'
  nntrainer_common_sources += 'cuda_buffer_manager.cpp'
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

# Install headers
install_headers(nntrainer_headers,
  install_dir: nntrainer_includedir,
)
