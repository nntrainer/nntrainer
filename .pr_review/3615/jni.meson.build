and_conf = configuration_data()

flags = []
arm_march_flag = ''
if get_option('debug')
    flags += '-g'
endif

# Architecture flags: applied based on platform (decoupled from enable-fp16)
if get_option('platform') == 'android'
  arm_march_flag = get_option('arm-march')
  if arm_march_flag == ''
    # Fallback: derive from arm-arch (for builds without --arm-arch argument)
    arm_arch_opt = get_option('arm-arch')
    if arm_arch_opt == 'armv9.2-a'
      arm_march_flag = '-march=armv9.2-a+fp16+sve2+sme'
    elif arm_arch_opt == 'armv8.2-a' or arm_arch_opt == 'auto'
      arm_march_flag = '-march=armv8.2-a+fp16+dotprod+i8mm'
    elif arm_arch_opt == 'armv7l'
      arm_march_flag = '-march=armv7-a+neon'
    endif
  endif
  if arm_march_flag != ''
    flags += arm_march_flag
  endif
endif

flags += '-O@0@'.format(get_option('optimization'))
message('compile flags are:' + ' '.join(flags))

flags = extra_defines + flags


and_conf.set('MESON_CFLAGS', ' '.join(flags))
and_conf.set('MESON_CXXFLAGS', ' '.join(flags))
and_conf.set('MESON_NNTRAINER_SRCS', ' '.join(nntrainer_sources))
and_conf.set('MESON_NNTRAINER_INCS', ' '.join(nntrainer_inc_abs))
and_conf.set('MESON_CCAPI_NNTRAINER_SRCS', ' '.join(ccapi_src))
and_conf.set('MESON_CCAPI_NNTRAINER_INCS', ' '.join(ccapi_inc_abs))
and_conf.set('MESON_ARM_MARCH', arm_march_flag)

and_conf.set('VERSION_MAJOR', nntrainer_version_split[0])
and_conf.set('VERSION_MINOR', nntrainer_version_split[1])
and_conf.set('VERSION_MICRO', nntrainer_version_split[2])

if get_option('enable-capi').enabled()
   and_conf.set('MESON_CAPI_NNTRAINER_SRCS', ' '.join(capi_src))
   and_conf.set('MESON_CAPI_NNTRAINER_INCS', ' '.join(capi_inc_abs))
else
   and_conf.set('MESON_CAPI_NNTRAINER_SRCS', ' '.join([]))
   and_conf.set('MESON_CAPI_NNTRAINER_INCS', ' '.join([]))   
endif

if iniparser_dep.found()
  and_conf.set('MESON_INIPARSER_ROOT', iniparser_root)
endif

if get_option('enable-ruy')
  if ruy_dep.found()
    and_conf.set('MESON_RUY_ROOT', ruy_root)
  endif
endif

if get_option('enable-opencl')
  # Set OpenCL block markers to include the OpenCL sections
  and_conf.set('MESON_ENABLE_OPENCL', 1) 
  if clblast_dep.found()
    and_conf.set('MESON_CLBLAST_ROOT', clblast_root)
    and_conf.set('MESON_CL_ROOT', opencl_root)
  endif
else
  # Disable OpenCL
  and_conf.set('MESON_ENABLE_OPENCL', 0)
  # No OpenCL libraries needed
  and_conf.set('MESON_CLBLAST_ROOT', '')
  and_conf.set('MESON_CL_ROOT', '')
endif

if tflite_dep.found()
  and_conf.set('MESON_HAS_TFLITE', 1)
  and_conf.set('MESON_TFLITE_ROOT', tflite_root)
else
  and_conf.set('MESON_HAS_TFLITE', 0)
endif

# -DML_API_COMMON=[01]
and_conf.set('ML_API_COMMON', ml_api_common_flag)

if blas_dep.found()
  and_conf.set('MESON_BLAS_ROOT', blas_root)
else
  error('blas is needed for the android build')
endif

if ml_api_common_dep.found()
  and_conf.set('MESON_ML_API_COMMON_ROOT', ml_api_common_root)
else
  error('ml api common dep is needed for the android build')
endif

configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)

install_data(sources: 'Android-prebuilt.mk', rename: 'Android.mk', install_dir: nntrainer_prefix)

configure_file(input: 'Application.mk', output: 'Application.mk',
  copy: true
)

# below is list of file that meson acknowledges this is the output of custom target so that
# this has to be installed by meson
outputs= [
  'arm64-v8a'
]


ndk_build = find_program('ndk-build', required : true)

ndk_args = {
  'NDK_LIBS_OUT': meson.current_build_dir(),
}
num_threads = run_command(nproc_prog, check: true).stdout().strip()

thread_opt_flag = '-j' + num_threads

ndk_additional_flags = [thread_opt_flag]

ndk_build_command = [ndk_build]
foreach key, val : ndk_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach
ndk_build_command += ndk_additional_flags

ndk_depends = []
ndk_input = []
foreach src: nntrainer_sources
  ndk_input += files(src)
endforeach

if get_option('enable-tflite-interpreter')
  ndk_depends += flat_header
endif

ndk_result = custom_target('ndk-build',
                           input: ndk_input,
                           output: outputs,
                           build_by_default: true,
                           depends: ndk_depends,
                           install: true,
                           install_dir: nntrainer_libdir,
                           command: [ndk_build_command])

