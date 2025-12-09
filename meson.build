project('nntrainer', 'c', 'cpp',
  version: '0.6.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

cmake_build_type = 'Release'
if get_option('buildtype') == 'debug'
  cmake_build_type = 'Debug'
endif

common_win_subproject_cmake_defines = {
  'CMAKE_MSVC_RUNTIME_LIBRARY': 'MultiThreaded',
}

# Set version info
nntrainer_version = meson.project_version()
nntrainer_version_split = nntrainer_version.split('.')

# CMake module
cmake = import('cmake')
fs = import('fs')

extra_defines = ['-DMIN_CPP_VERSION=201703L']

add_project_arguments('-DVERSION="' + nntrainer_version + '"', language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MAJOR=' + nntrainer_version_split[0], language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MINOR=' + nntrainer_version_split[1], language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MICRO=' + nntrainer_version_split[2], language: ['c', 'cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
cxx_compiler_id = cxx.get_id()

powershell_prog = find_program('powershell', required: (build_machine.system() == 'windows'))

# Obtains number of cores
nproc_prog = find_program('nproc',
                          required: (get_option('platform') == 'android') and (build_machine.system() != 'windows'))

prog_win_cmd = find_program('cmd.exe', required: build_machine.system() == 'windows')

if get_option('platform') == 'tizen'
  # Pass __TIZEN__ to the compiler
  add_project_arguments('-D__TIZEN__=1', language:['c','cpp'])
  add_project_arguments('-DTIZENVERSION=@0@'.format(get_option('tizen-version-major')), language: ['c', 'cpp'])
  add_project_arguments('-DTIZENVERSIONMINOR=@0@'.format(get_option('tizen-version-minor')), language: ['c', 'cpp'])

  if get_option('enable-tizen-feature-check')
    add_project_arguments('-D__FEATURE_CHECK_SUPPORT__', language: ['c', 'cpp'])
  endif
endif

if get_option('enable-transformer')
  add_project_arguments('-DENABLE_TRANSFORMER=1', language: ['c', 'cpp'])
endif

warning_flags = [
  '-Wredundant-decls',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Winit-self',
  '-Waddress',
  '-Wvla',
  '-Wpointer-arith',
  '-Wno-error=varargs',
  '-Wno-error=ignored-attributes',
  '-Wno-error=sign-compare',
  '-Wdefaulted-function-deleted',
  '-ftree-vectorize',
  '-Wno-unused-variable',
  '-Wno-comment',
  '-Wno-ignored-attributes',
  '-Wno-sign-compare'
]

if cxx_compiler_id == 'clang'
  warning_flags += '-Wno-uninitialized'
  warning_flags += '-Wno-format-nonliteral'
  warning_flags += '-Wno-varargs'
  warning_flags += '-Wno-missing-braces'
else
  warning_flags += '-Wno-maybe-uninitialized'
endif

warning_c_flags = [
  '-Wmissing-declarations',
  '-Wmissing-include-dirs',
  '-Wmissing-prototypes',
  '-Wnested-externs',
  '-Waggregate-return',
  '-Wold-style-definition',
  '-Wdeclaration-after-statement',
  '-Wno-error=varargs'
]

arch = host_machine.cpu_family()

target = target_machine.cpu_family()


subprojects_dir_relative = 'subprojects'
subprojects_dir_absolute = meson.current_source_dir() / subprojects_dir_relative

if get_option('platform') == 'windows'
  if arch == 'arm' or arch == 'aarch64'
  windows_resource_dir_relative = 'nntrainer-windows-resource' / 'arm64'
  windows_resource_dir_absolute = meson.current_source_dir() / windows_resource_dir_relative
  else
  windows_resource_dir_relative = 'nntrainer-windows-resource' / 'x64'
  windows_resource_dir_absolute = meson.current_source_dir() / windows_resource_dir_relative
  endif
endif

if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  message('Build for ARM architecture')
  if arch == 'arm'
    extra_defines += '-DARMV7=1'
  endif
elif arch == 'x86' or arch == 'x86_64'
  message('Build for X86 architecture')
  if cxx_compiler_id == 'msvc'
    add_project_arguments(['/arch:AVX2'], language: ['c','cpp'])
    # Note: C runtime library (/MT or /MD) is controlled by b_vscrt option
  else
    add_project_arguments(['-march=native'], language: ['c','cpp'])
    add_project_arguments(['-mavx2', '-mfma'], language: ['c','cpp'])
    message('-march=native added for AVX hardware acceleration.')
  endif
endif

if get_option('enable-npu')
     extra_defines += '-DENABLE_NPU=1'
endif

if get_option('mmap-read')
  if get_option('enable-transformer')
    message('mmap-read cannot be enabled with enable-transformer. mmap-read is turned off.')
    extra_defines += '-DMMAP_READ=0'
  else
    extra_defines += '-DMMAP_READ=1'
  endif
else
    extra_defines += '-DMMAP_READ=0'
endif

if get_option('enable-kernel-caching')
  extra_defines += '-DKERNEL_CACHE=1'
else
  extra_defines += '-DKERNEL_CACHE=0'
endif

if get_option('enable-fp16')
   if get_option('platform') == 'android'
     add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
     extra_defines += '-DENABLE_FP16=1'
     extra_defines += '-DUSE__FP16=1'
   elif arch == 'aarch64'
     ## About FP16 in GCC (from GCC-9.1 manual)
     # https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Half-Precision.html
     # On ARM and AArch64 targets, GCC supports half-precision (16-bit) floating point
     #   via the __fp16 type defined in the ARM C Language Extensions.
     # On ARM systems, you must enable this type explicitly with the -mfp16-format
     #   command-line option in order to use it.
     ## About FP16-SIMD in aarch64
     # FP16-SIMD is supported since armv8.2. If you enable this forcibly, it won't be
     # comaptible with armv8.0 machines.
     if cxx.has_argument('-mfp16-format=ieee')
       add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
       add_project_arguments('-march=armv8.2-a+fp16', language: ['c', 'cpp'])
     else
       message ('The compiler does not support -mfp16-format=ieee. However, according to https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Half-Precision.html, gcc may use IEEE fp16 anyway. Thus, we will proceed without the option for FP16 support.')
     endif
     extra_defines += '-DENABLE_FP16=1'
     extra_defines += '-DUSE__FP16=1'
   elif arch == 'arm'
     ## About FP16-SIMD in arm
     # FP16-SIMD is supported since armv8.2.
     # Thus, even if fp16 is force-enabled, NEON is off.
     if cxx.has_argument('-mfp16-format=ieee')
       add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
       extra_defines += '-DENABLE_FP16=1'
       extra_defines += '-DUSE__FP16=1'
     else
       error ('The compiler does not support -mfp16-format=ieee')
     endif
   elif arch == 'x86_64'
     if cc.version().version_compare('>=12.1.0')
       message ('Float16 for x86_64 enabled. Modern gcc-x64 generally supports float16 with _Float16.')
       extra_defines += '-DENABLE_FP16=1'
     else
       warning ('Float16 for x86_64 enabled. However, software emulation is applied for fp16, making it slower and inconsistent. Use GCC 12+ for FP16 support. This build will probably fail unless you bring a compiler that supports fp16 for x64.')
     endif
   elif arch == 'riscv64'
     error ('RISCV64 RVV support and fp16 support is not yet implemented.')
   else
     error ('FP16 support for this arch is not yet implemented.')
   endif  
endif

if get_option('enable-mmap')
  message ('MMAP enabled')
  extra_defines += '-DUSE_MMAP=1'
endif

dummy_dep = dependency('', required: false)

if get_option('enable-opencl')
  if get_option('platform') == 'android'
  endif

  message ('OpenCL build is now enabled and will function only if an OpenCL-supported GPU is available. Using CLBlast as the OpenCL backend.')
  extra_defines += '-DENABLE_OPENCL=1'
  extra_defines += '-DCL_TARGET_OPENCL_VERSION=200'
  clblast_options = cmake.subproject_options()

  if get_option('platform') == 'android'
    message('preparing opencl')
    run_command([meson.source_root() / 'jni' / 'prepare_opencl.sh', meson.build_root()], check: true)
    clblast_root = subprojects_dir_absolute / 'CLBlast'
    opencl_root = meson.build_root() / 'opencl'
    clblast_dep = declare_dependency()
  else 
    clblast_options = cmake.subproject_options()
    clblast_options.add_cmake_defines({'CMAKE_POLICY_VERSION_MINIMUM': '3.10'})
    clblast_options.add_cmake_defines({'CMAKE_CXX_STANDARD': '17'})
    clblast_options.add_cmake_defines({'TUNERS': false})
    clblast_options.add_cmake_defines({'CMAKE_BUILD_TYPE': cmake_build_type})
    if get_option('default_library') == 'static'
      clblast_options.add_cmake_defines({'OVERRIDE_MSVC_FLAGS_TO_MT': false})
      clblast_options.add_cmake_defines({'BUILD_SHARED_LIBS': false})
    endif
    clblast_proj = cmake.subproject('clblast', options: clblast_options, required: true)
    clblast_dep = clblast_proj.dependency('clblast')
  endif
endif

if get_option('opencl-kernel-path') != ''
  message ('OpenCL kernel path set to: @0@'.format(get_option('opencl-kernel-path')))
  extra_defines += '-DOPENCL_KERNEL_PATH=@0@'.format(get_option('opencl-kernel-path'))
endif

if get_option('enable-biqgemm')
  # In order to enable biqgemm, BiQGEMM lib, which is header-only library, is required.
  # This meson tries to find its installation.
  #   1. Checking 'prefix'/'includedir'/BiQGEMM
  #   2. Checking path meson-option specifies
  biqgemm_path = join_paths(get_option('prefix'), get_option('includedir'), 'BiQGEMM')
  if cxx.has_header('BiQGEMM.h', args: '-I'+biqgemm_path)
      message('[lib:biqgemm] biqgemm header is found successfully')
      extra_defines += '-DENABLE_BIQGEMM=1'
      biqgemm_inc = include_directories(biqgemm_path)
  else
      # relative path from biqgemm is assumed
      biqgemm_path = get_option('biqgemm-path')
      message('[lib:biqgemm] fallback: finding biqgemm from user-path :' + biqgemm_path)
      if fs.is_dir(biqgemm_path) 
        message('[lib:biqgemm] biqgemm header is found successfully')
        extra_defines += '-DENABLE_BIQGEMM=1'
        biqgemm_inc = include_directories(biqgemm_path)
      else
        error ('BiQGEMM cannot be enabled without BiQGEMM library.')
      endif
  endif
endif # end of enable-biqgemm

if cc.get_id() != 'msvc' and cxx.get_id() != 'msvc'
  foreach extra_arg : warning_flags
    if cc.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'c')
    endif
    if cxx.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'cpp')
    endif
  endforeach

  foreach extra_arg : warning_c_flags
    if cc.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'c')
    endif
  endforeach
endif

# Set install path
nntrainer_prefix = get_option('prefix')
if get_option('platform') != 'android'
  nntrainer_libdir = nntrainer_prefix / get_option('libdir')
  nntrainer_bindir = nntrainer_prefix / get_option('bindir')
  nntrainer_includedir = nntrainer_prefix / get_option('includedir') / 'nntrainer'
  nntrainer_confdir = get_option('sysconfdir')
  application_install_dir = nntrainer_bindir / 'applications'
  nntrainer_fsudir = '/tmp'
else
  nntrainer_prefix = meson.build_root() / 'android_build_result'
  # @todo arch has to be option
  nntrainer_libdir = nntrainer_prefix / 'lib'
  nntrainer_includedir = nntrainer_prefix / 'include' / 'nntrainer'
  nntrainer_bindir = nntrainer_prefix / 'bin'
  nntrainer_confdir = nntrainer_prefix / 'conf'
  application_install_dir = nntrainer_prefix / 'examples'
  nntrainer_fsudir = '/data/local/tmp'
endif

# handle fsu options
if get_option('enable-fsu')
  nntrainer_enable_fsu = 'true'
else
  nntrainer_enable_fsu = 'false'
endif

if get_option('fsu-path') != ''
  nntrainer_fsudir = get_option('fsu-path')
endif

# handle resources
nntrainer_resdir = meson.build_root() / 'res'

if get_option('platform') == 'windows'
  nntrainer_resdir_win = nntrainer_resdir.replace('/', '\\')
  if not fs.exists (nntrainer_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntrainer_resdir_win], check: true)
  endif
else
  run_command(['mkdir', '-p', nntrainer_resdir], check: true)
endif

if get_option('install-app')
# add a script to install resources from installs to application_install_dir
meson.add_install_script(
  'sh', '-c', 'cp -r @0@ ${DESTDIR}@1@'.format(nntrainer_resdir, application_install_dir)
)
endif

# Set default configuration
nntrainer_conf = configuration_data()
nntrainer_conf.set('VERSION', meson.project_version())
nntrainer_conf.set('PREFIX', nntrainer_prefix)
nntrainer_conf.set('EXEC_PREFIX', nntrainer_bindir)
nntrainer_conf.set('LIB_INSTALL_DIR', nntrainer_libdir)
nntrainer_conf.set('PLUGIN_INSTALL_PREFIX', nntrainer_libdir / 'nntrainer')
nntrainer_conf.set('INCLUDE_INSTALL_DIR', nntrainer_includedir / '..')
nntrainer_conf.set('FLASH_STORAGE_UTILIZATION', nntrainer_enable_fsu)
nntrainer_conf.set('FLASH_STORAGE_UTILIZATION_PATH', nntrainer_fsudir)

ml_api_common_flag = '-DML_API_COMMON=0'

# if ml-api-support is disabled, enable dummy common api interfaces and disable related dependencies.
ml_api_common_dep = dependency(get_option('capi-ml-common-actual'), required : get_option('ml-api-support').enabled())
nnstreamer_capi_dep = dummy_dep
if (ml_api_common_dep.found())
  nnstreamer_capi_dep = dependency(get_option('capi-ml-inference-actual'), required : get_option('ml-api-support').enabled())
  if (nnstreamer_capi_dep.found())
    nntrainer_conf.set('CAPI_ML_COMMON_DEP', get_option('capi-ml-common-actual'))
    extra_defines += '-DML_API_COMMON=1'
    ml_api_common_flag = '-DML_API_COMMON=1'
    extra_defines += '-DNNSTREAMER_AVAILABLE=1'
    # accessing this variable when dep_.not_found() remains hard error on purpose
    supported_nnstreamer_capi = nnstreamer_capi_dep.version().version_compare('>=1.7.0')
    if not supported_nnstreamer_capi
      extra_defines += '-DUNSUPPORTED_NNSTREAMER=1'
      warning('capi-nnstreamer version is too old, we do not know if it works with older nnstreamer version')
    endif
  else
    # if nnstreamer_capi is not there and ml-api-support is "auto", disable it.
    message ('ml-api-support is disabled although capi-ml-api-common is found: capi-ml-api-inference is not available and ml-api-support is configured to be auto')
    nntrainer_conf.set('CAPI_ML_COMMON_DEP', '')
    extra_defines += '-DML_API_COMMON=0'
  endif
else
  nntrainer_conf.set('CAPI_ML_COMMON_DEP', '')
  extra_defines += '-DML_API_COMMON=0'
endif
blas_dep = dummy_dep
# Dependencies
if get_option('enable-cublas')
  extra_defines += '-DUSE_CUBLAS=1'
endif

if get_option('enable-blas')
  extra_defines += '-DUSE_BLAS=1'

  if get_option('platform') == 'android'
    message('preparing blas')
    run_command([meson.source_root() / 'jni' / 'prepare_openblas.sh', meson.build_root()], check: true)
    blas_root = meson.build_root() / 'openblas'
    blas_dep = declare_dependency(include_directories: [ 'openblas/include' ])
  elif get_option('platform') == 'windows'
    openblas_lib_dir_absolute = windows_resource_dir_absolute / 'OpenBLAS'
    openblas_lib_dir_absolute_win = openblas_lib_dir_absolute.replace('/', '\\')
    run_command(['xcopy', '/C', '/Y', openblas_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    openblas_dir_relative = subprojects_dir_relative / 'OpenBLAS'
    openblas_lib_dir_relative = windows_resource_dir_relative / 'OpenBLAS'
    openblas_include = include_directories(openblas_dir_relative, openblas_lib_dir_relative)
    blas_dep = declare_dependency(include_directories: openblas_include, link_args:['openblas.lib'])
  else
    blas_dep = dependency('openblas', required : false)

    if not blas_dep.found()
      message('Cannot find OpenBLAS dependency in system')
      openblas_dir_relative = subprojects_dir_relative / 'OpenBLAS'
      openblas_dir_absolute = meson.current_source_dir() / openblas_dir_relative
      openblas_build_dir_relative = openblas_dir_relative / 'build'
      openblas_build_dir_absolute = meson.current_source_dir() / openblas_build_dir_relative

      if fs.exists (openblas_build_dir_absolute)
        message('Skip compiling OpenBLAS (using previous build)')
      else
        message('Setup OpenBLAS')
        openblas_setup_command = ['cmake', '-B', openblas_build_dir_absolute, '-DNUM_THREADS=' + get_option('openblas-num-threads').to_string(), '-DBUILD_TESTING=False', '-DBUILD_BENCHMARKS=False', '-DBUILD_WITHOUT_LAPACK=True', '-DNOFORTRAN=True', '-DCMAKE_BUILD_TYPE='+cmake_build_type]
        openblas_setup_command += '-DBUILD_SHARED_LIBS=True'

        if cxx_compiler_id == 'clang'
          openblas_setup_command += '-G Ninja'
          openblas_setup_command += '-DCMAKE_C_COMPILER=clang'
          openblas_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
        endif

        openblas_setup_command += openblas_dir_absolute

        run_command(openblas_setup_command, check: true)

        message('Compile OpenBLAS')
        openblas_compile_command = ['cmake' , '--build', openblas_build_dir_absolute, '--config', cmake_build_type, '-j']
        run_command(openblas_compile_command, check: true)
      endif

      openblas_lib = openblas_build_dir_absolute / 'lib' / 'libopenblas.so'
      openblas_include = include_directories(openblas_dir_relative, openblas_build_dir_relative)
      blas_dep = declare_dependency(include_directories: openblas_include, link_args:[openblas_lib])
    else
      message('OpenBLAS dependency found in system')
    endif
  endif

  if blas_dep.found()
    if get_option('openblas-num-threads') > 0
      extra_defines += '-DBLAS_NUM_THREADS=@0@'.format(get_option('openblas-num-threads'))
      message('set openblas num threads=@0@'.format(get_option('openblas-num-threads')))
    endif
  endif
  extra_defines += '-DHGEMM_EXPERIMENTAL_KERNEL=@0@'.format(get_option('hgemm-experimental-kernel'))
endif

extra_defines += '-DNNTR_NUM_THREADS=@0@'.format(get_option('nntr-num-threads'))
message('set nntrainer num threads=@0@'.format(get_option('nntr-num-threads')))
openmp_dep = dummy_dep
if get_option('enable-openmp')
 # For Windows on x64, use OpenMP 5.0 instead of the OpenMP 2.0 standard
  openmp_win_found = false
  if cxx_compiler_id == 'msvc' and (arch == 'x86' or arch == 'x86_64')
    omp_lib_subdir = 'x64'
    omp_lib_name = 'libiomp5md'
    openmp_win_found = true
  elif cxx_compiler_id == 'clang-cl' and (arch == 'aarch64' or arch == 'arm64')
    omp_lib_subdir = 'arm64'
    omp_lib_name = 'libomp'
    openmp_win_found = true
  endif

  if openmp_win_found
    libiomp_root_opt = get_option('libiomp_root')
    libiomp_root = join_paths(meson.project_source_root(), libiomp_root_opt)
    omp_libdir = libiomp_root / omp_lib_subdir
    omp_incdir = join_paths(libiomp_root_opt, 'include')

    omp_libdir_win = omp_libdir.replace('/', '\\')
    run_command(['xcopy', '/C', '/Y', omp_libdir_win, meson.current_build_dir()], check: true)

    add_project_arguments('/openmp', language: ['cpp','c'])
    cpp = meson.get_compiler('cpp')
    libiomp = cpp.find_library(omp_lib_name, dirs: omp_libdir, required: true)

    openmp_dep = declare_dependency(
      include_directories : include_directories(omp_incdir),
      compile_args : ['/openmp'],
      link_args : ['/NODEFAULTLIB:vcomp'],
      dependencies : [libiomp])

    message('set openmp with @0@ for windows @1@-----------------'.format(omp_lib_name, omp_lib_subdir))
  else
    openmp_dep = dependency('openmp')
  endif
  if get_option('omp-num-threads') > 0
    extra_defines += '-DOMP_NUM_THREADS=@0@'.format(get_option('omp-num-threads'))
    message('set nntrainer omp threads=@0@'.format(get_option('omp-num-threads')))
  endif
endif

if get_option('enable-profile')
  extra_defines += '-DPROFILE=1'
endif

if get_option('enable-trace')
  extra_defines += '-DTRACE=1'
endif

if get_option('enable-debug')
  extra_defines += '-DDEBUG=1'
endif

if get_option('use_gym')
  extra_defines += '-DUSE_GYM=1'
endif

if get_option('enable-logging')
  extra_defines += '-D__LOGGING__=1'
endif

if get_option('enable-benchmarks')
  if get_option('platform') == 'windows'
    benchmark_lib_dir_absolute = windows_resource_dir_absolute / 'benchmark'
    benchmark_lib_dir_absolute_win = benchmark_lib_dir_absolute.replace('/', '\\')
    run_command(['xcopy', '/C', '/Y', benchmark_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    benchmark_dir_relative = subprojects_dir_relative / 'benchmark'
    benchmark_include = include_directories(benchmark_dir_relative / 'include')
    benchmark_dep = declare_dependency(include_directories: benchmark_include, link_args:['benchmark.lib'])
  else
    benchmark_dep = dependency('benchmark', static : true, main : false, required : false)
    if not benchmark_dep.found()
      benchmark_options = cmake.subproject_options()
      benchmark_options.add_cmake_defines({'BENCHMARK_ENABLE_TESTING': false})
      benchmark_subproject = cmake.subproject('benchmark', options: benchmark_options, required: true)
      benchmark_dep = benchmark_subproject.dependency('benchmark')
      message('benchmar targets')
      message(benchmark_subproject.target_list())
    endif
  endif
endif

if get_option('platform') == 'windows'
    googletest_lib_dir_absolute = windows_resource_dir_absolute / 'googletest'
    googletest_lib_dir_absolute_win = googletest_lib_dir_absolute.replace('/', '\\')
    run_command(['xcopy', '/C', '/Y', googletest_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    googletest_dir_relative = subprojects_dir_relative / 'googletest'
    gtest_include = include_directories(googletest_dir_relative / 'googletest' / 'include')
    gmock_include = include_directories(googletest_dir_relative / 'googlemock' / 'include')
    gmock_dep = declare_dependency(include_directories: gmock_include, link_args:['gmock.lib'])
    gtest_dep = declare_dependency(include_directories: gtest_include, link_args:['gtest.lib'])
    gtest_main_dep = declare_dependency(include_directories: gtest_include, link_args:['gtest_main.lib', 'gtest.lib'])
else
  gmock_dep = dependency('gmock', static: true, main: false, required: false)
  gtest_dep = dependency('gtest', static: true, main: false, required: false)
  gtest_main_dep = dependency('gtest', static: true, main: true, required: false)

  if not gmock_dep.found() or not gtest_dep.found() or not gtest_main_dep.found()
    googletest_options = cmake.subproject_options()
    if host_machine.system() != 'windows'
      googletest_options.add_cmake_defines({'CMAKE_POSITION_INDEPENDENT_CODE': true})
    endif
    googletest_options.add_cmake_defines({'BUILD_SHARED_LIBS': false})
    googletest_subproject = cmake.subproject('googletest', options: googletest_options, required: true)
    gmock_dep = googletest_subproject.dependency('gmock')
    gtest_dep = googletest_subproject.dependency('gtest')
    gtest_main_dep = [googletest_subproject.dependency('gtest_main')]
    # meson provided dependency gtest_main_dep is sum of [gtest_dep, gtest_main_dep]
    # This difference between cmake and meson dependency is exposed on linux
    gtest_main_dep += gtest_dep
  endif
endif

if get_option('enable-test') # and get_option('platform') != 'android'
  extra_defines += '-DENABLE_TEST=1'
  if gtest_dep.version().version_compare('<1.10.0')
     extra_defines += '-DGTEST_BACKPORT=1'
  endif
  test_timeout = get_option('test-timeout')
endif

if get_option('reduce-tolerance')
  extra_defines += '-DREDUCE_TOLERANCE=1'
endif

libm_dep = cxx.find_library('m', required: false) # cmath library
libdl_dep = cxx.find_library('dl', required: false) # DL library

thread_dep = dependency('threads') # pthread for tensorflow-lite

if get_option('platform') == 'android'
  iniparser_root = subprojects_dir_absolute / 'iniparser'
  iniparser_dep = declare_dependency()
else
  iniparser_dep = dependency('iniparser', required : false, version : '>=3.2') # iniparser

  if not iniparser_dep.found()
    iniparser_options = cmake.subproject_options()
    iniparser_options.add_cmake_defines({'BUILD_DOCS': false})
    iniparser_options.add_cmake_defines({'BUILD_EXAMPLES': false})
    iniparser_options.add_cmake_defines({'BUILD_TESTING': false})
    iniparser_options.add_cmake_defines({'BUILD_SHARED_LIBS': false})
    iniparser_options.add_cmake_defines({'CMAKE_BUILD_TYPE': cmake_build_type})
    if get_option('platform') == 'windows'
      iniparser_options.add_cmake_defines(common_win_subproject_cmake_defines)
      if arch == 'aarch64' or arch == 'arm64'
        iniparser_options.add_cmake_defines({'CMAKE_SYSTEM_PROCESSOR': 'ARM64'})
        iniparser_options.add_cmake_defines({'CMAKE_C_FLAGS': '--target=aarch64-pc-windows-msvc'})
        iniparser_options.add_cmake_defines({'CMAKE_CXX_FLAGS': '--target=aarch64-pc-windows-msvc'})
        iniparser_options.add_cmake_defines({'CMAKE_STATIC_LINKER_FLAGS': '/machine:ARM64'})
        iniparser_options.add_cmake_defines({'CMAKE_SHARED_LINKER_FLAGS': '/machine:ARM64'})
        iniparser_options.add_cmake_defines({'CMAKE_EXE_LINKER_FLAGS': '/machine:ARM64'})
      endif
    endif
    iniparser_subproject = cmake.subproject('iniparser', options: iniparser_options, required: true)
    iniparser_dep = iniparser_subproject.dependency('iniparser-static')
  endif
endif

# Configure the Ruy project (CMake)
ruy_dep = dummy_dep
if get_option('enable-ruy')
if get_option('platform') == 'android'
  ruy_root = subprojects_dir_absolute / 'ruy'
  ruy_dep = declare_dependency()
elif get_option('platform') == 'windows'
  ruy_dir_relative = subprojects_dir_relative / 'ruy'
  ruy_include = include_directories(ruy_dir_relative)
  ruy_dep = declare_dependency(include_directories: ruy_include)
else
  ruy_options = cmake.subproject_options()
  ruy_options.add_cmake_defines({'RUY_MINIMAL_BUILD': true})
  ruy_options.add_cmake_defines({'CMAKE_POLICY_VERSION_MINIMUM': '3.5'})

  ruy_flags = [
    '-Wno-error=unused-result',
    '-Wno-error=comment',
    '-Wno-error=unknown-pragmas',
    '-Wno-error=unused-function'
    ]

  if cxx_compiler_id == 'clang'
    ruy_flags += '-Wno-error=uninitialized'
  else
    ruy_flags += '-Wno-error=maybe-uninitialized'
    if cxx.has_argument('-Wno-error=stringop-overread') and cxx.has_argument('-Wno-error=array-parameter')
      ruy_flags += '-Wno-error=stringop-overread'
      ruy_flags += '-Wno-error=array-parameter'
    endif
  endif
  
  ruy_options.append_compile_args('c', '-Wno-implicit-function-declaration')
  ruy_options.append_compile_args('c', ruy_flags)
  ruy_options.append_compile_args('cpp', ruy_flags)

  ruy_proj = cmake.subproject('ruy', options: ruy_options, required: true)
  ruy_dep = ruy_proj.dependency('ruy').partial_dependency(includes: true)
endif
endif # enable ruy

if get_option('platform') == 'android'
  message('preparing ml api')
  run_command([meson.source_root() / 'jni' / 'prepare_ml-api.sh', meson.build_root() / 'ml-api-inference'], check: true)
  ml_api_common_root = meson.build_root() / 'ml-api-inference'
  ml_api_inc = ml_api_common_root / 'include'
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'ml-api-common.h', nntrainer_includedir)
  )
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'tizen_error.h', nntrainer_includedir)
  )
  ml_api_common_dep = declare_dependency(include_directories: ['ml-api-inference/include'])
endif

if get_option('enable-nnstreamer-backbone') and get_option('platform') != 'android'
  extra_defines += '-DENABLE_NNSTREAMER_BACKBONE=1'
endif

tflite_dep = dummy_dep

if get_option('platform') != 'android'
  tflite_dep = dependency('tensorflow2-lite', required: false)
else
  if get_option('enable-tflite-backbone') or get_option('enable-tflite-interpreter')
    tflite_version = '2.18.0'
    message('preparing tflite ' + tflite_version + ', because either tflite backbone or interpreter is enabled')
    run_command(join_paths(meson.source_root(), 'jni', 'prepare_tflite.sh'), tflite_version, meson.build_root(), check: true)
    tflite_root = join_paths(meson.build_root(), 'tensorflow-' + tflite_version, 'tensorflow-lite')
    tflite_dep = declare_dependency(include_directories: [ join_paths('tensorflow-' + tflite_version, 'tensorflow-lite', 'include') ])
  endif
endif

if get_option('enable-tflite-backbone')
  extra_defines += '-DENABLE_TFLITE_BACKBONE=1'
endif

if get_option('enable-tflite-interpreter')
  extra_defines += '-DENABLE_TFLITE_INTERPRETER=1'
endif

if get_option('enable-onnx-interpreter')
  extra_defines += '-DENABLE_ONNX_INTERPRETER=1'
endif

opencv_dep = dummy_dep

if get_option('platform') != 'android'
  opencv_dep = dependency('opencv', required: false)
  if not opencv_dep.found()
    opencv_dep = dependency('opencv4', required: false)
    if not opencv_dep.found()
      opencv_dep = dependency('opencv3', required: false)
    endif
  endif
  if opencv_dep.found()
    extra_defines += '-DENABLE_DATA_AUGMENTATION_OPENCV=1'
  endif
endif
flatc_prog = find_program('flatc', required: false)

# Install .pc
configure_file(input: 'nntrainer.pc.in', output: 'nntrainer.pc',
  install_dir: nntrainer_libdir / 'pkgconfig',
  configuration: nntrainer_conf
)

# Install conf
configure_file(
  input: 'nntrainer.ini.in',
  output: 'nntrainer.ini',
  install_dir: nntrainer_confdir,
  configuration: nntrainer_conf
)
nntrainer_conf_abs_path = get_option('prefix') / nntrainer_confdir / 'nntrainer.ini'
message('NNTRAINER_CONF_PATH=@0@'.format(nntrainer_conf_abs_path))

if get_option('platform') != 'android'
  extra_defines += '-DNNTRAINER_CONF_PATH="@0@"'.format(nntrainer_conf_abs_path)
endif

message('extra defines are:' + ' '.join(extra_defines))
foreach defs: extra_defines
  add_project_arguments(defs, language: ['c', 'cpp'])
endforeach

# Build nntrainer
subdir('nntrainer')

enable_capi = false
enable_ccapi = false
# Build api
subdir('api')

if get_option('enable-test')
  if get_option('platform') == 'android'
    warning('test is not supported in android build, test skipped')
  else
    if gtest_dep.found()
      subdir('test')
    else
      error('test enabled but gtest not found')
    endif
  endif
endif

if get_option('enable-app')
  if get_option('platform') == 'android'
    warning('android app is not supported for now, building app skipped')
  else
    if host_machine.system() != 'windows'
      if get_option('enable-tflite-backbone') and not tflite_dep.found()
        error('Tensorflow-Lite dependency not found')
      endif
    endif
    subdir('Applications')
  endif
endif

if get_option('platform') != 'android' and get_option('platform') != 'windows' and get_option('enable-nnstreamer-backbone')
  nnstreamer_dep = dependency('nnstreamer')
  message('building nnstreamer')
  subdir('nnstreamer')
else
  warning('android nnstreamer-filter and nnstreamer-trainer are not yet supported, building them is skipped')
endif

if get_option('platform') == 'android'
  subdir('jni')
endif

if get_option('platform') != 'none'
  message('building for ' + get_option('platform'))
endif

if get_option('enable-benchmarks')
  subdir('benchmarks')
endif
