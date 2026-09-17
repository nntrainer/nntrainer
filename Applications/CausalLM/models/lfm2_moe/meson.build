lfm2_moe_inc = include_directories('.')

# --- Model classes (compiled into libcausallm) ---
lfm2_moe_src = [
    meson.current_source_dir() / 'lfm2_moe_causallm.cpp',
    meson.current_source_dir() / 'lfm2_slim_moe_causallm.cpp',
]

# --- MoE layers (each built as its own shared library) ---
lfm2_moe_layer = shared_library(
    'lfm2_moe_layer',
    [meson.current_source_dir() / 'lfm2_moe_layer.cpp'],
    include_directories: [causallm_layer_inc, lfm2_moe_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
lfm2_moe_layer_dep = declare_dependency(
    link_with: lfm2_moe_layer,
    include_directories: lfm2_moe_inc
)

lfm2_slim_moe_layer = shared_library(
    'lfm2_slim_moe_layer',
    [meson.current_source_dir() / 'lfm2_moe_layer_fsu.cpp'],
    include_directories: [causallm_layer_inc, lfm2_moe_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
lfm2_slim_moe_layer_dep = declare_dependency(
    link_with: lfm2_slim_moe_layer,
    include_directories: lfm2_moe_inc
)

causallm_src += lfm2_moe_src
causallm_inc += lfm2_moe_inc
causallm_layer_dependencies += [lfm2_moe_layer_dep, lfm2_slim_moe_layer_dep]

# Cached-Slim variant uses std::mutex/std::list caching; gated to non-windows
# (mirrors qwen3_cached_slim_moe).
if get_option('platform') != 'windows'
    lfm2_cached_slim_moe_layer = shared_library(
        'lfm2_cached_slim_moe_layer',
        [meson.current_source_dir() / 'lfm2_moe_layer_cached.cpp'],
        include_directories: [causallm_layer_inc, lfm2_moe_inc],
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
        install: true,
        install_dir: application_install_dir
    )
    lfm2_cached_slim_moe_layer_dep = declare_dependency(
        link_with: lfm2_cached_slim_moe_layer,
        include_directories: lfm2_moe_inc
    )
    causallm_src += [meson.current_source_dir() / 'lfm2_cached_slim_moe_causallm.cpp']
    causallm_layer_dependencies += [lfm2_cached_slim_moe_layer_dep]
endif
