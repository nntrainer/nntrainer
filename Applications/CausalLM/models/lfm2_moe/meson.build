causallm_inc += include_directories('.')

# Preserve exact router arithmetic independently of the fused model's flags.
lfm2_router_arm_cpp_args = []
if cxx.get_id() in ['gcc', 'clang', 'apple-clang']
    lfm2_router_arm_cpp_args += ['-fno-fast-math', '-ffp-contract=off']
endif
lfm2_router_arm_lib = static_library('lfm2_router_arm',
    'lfm2_router_arm.cpp',
    dependencies: [nntrainer_dep],
    cpp_args: lfm2_router_arm_cpp_args,
)
causallm_layer_dependencies += declare_dependency(link_with: lfm2_router_arm_lib)
