pi05_src = [
    meson.current_source_dir() / 'adarms_layer.cpp',
    meson.current_source_dir() / 'pi05_model.cpp',
]

if get_option('enable-cuda')
    pi05_src += meson.current_source_dir() / 'pi05_cuda.cpp'
endif

pi05_inc = include_directories('.')

causallm_src += pi05_src
causallm_inc += pi05_inc
