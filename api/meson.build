if get_option('enable-ccapi')
  enable_ccapi = true
  subdir('ccapi')
endif

# build_capi is computed in the top-level meson.build.
if build_capi
  if get_option('enable-capi').enabled() and not get_option('enable-ccapi')
    error('enable-ccapi must be set for capi as well')
  endif
  enable_capi = true
  subdir('capi')
endif
