# Backend-neutral LLM layers. Compiled straight into libnntrainer (the loss/
# subdir pattern), not as per-layer shared_library plugins. Add sources
# incrementally, one per layer.
llm_layer_sources = [
  'geglu_layer.cpp',
]

# NOTE: concrete layer headers are not part of the -devel surface. They stay out
# of nntrainer_headers, like every other concrete layer (fc_layer.h,
# layer_normalization_layer.h, the cl_layers/ headers); only base classes such as
# layer_devel.h and loss_layer.h are installed. Anything added to
# nntrainer_headers must also be listed in packaging/nntrainer.spec %files devel
# and debian/nntrainer-dev.install, or the RPM/deb builds fail on an
# installed-but-unpackaged file. The include path for this directory comes from
# nntrainer_inc in ../meson.build, so in-tree users still resolve <geglu_layer.h>.

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
