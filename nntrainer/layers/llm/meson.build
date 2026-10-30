# Backend-neutral LLM layers. Compiled straight into libnntrainer (the loss/
# subdir pattern), not as per-layer shared_library plugins. Each one owns shape
# and orchestration only and reaches its kernel through the tensor's ComputeOps
# table, so one class serves every backend. Add sources incrementally, one per
# layer.
llm_layer_sources = [
  'geglu_layer.cpp',
  'lm_head.cpp',
  'logit_softcapping.cpp',
  'qkv_layer.cpp',
  'scalar_multiply.cpp',
  'swiglu_layer.cpp',
  'tie_word_embedding.cpp',
  'htp_tie_word_embedding.cpp',
  'sigmoid_glu_layer.cpp',
  'sigmoid_add_layer.cpp',
]

# NOTE: concrete layer headers are not part of the -devel surface. They stay out
# of nntrainer_headers, like every other concrete layer (fc_layer.h,
# layer_normalization_layer.h, the cl_layers/ headers); only base classes such as
# layer_devel.h and loss_layer.h are installed. Anything added to
# nntrainer_headers must also be listed in packaging/nntrainer.spec %files devel
# and debian/nntrainer-dev.install, or the RPM/deb builds fail on an
# installed-but-unpackaged file. The include path for this directory comes from
# nntrainer_inc in ../meson.build, so in-tree users still resolve <geglu_layer.h>.
#
# layer_prof.h is the one exception. It is not a layer: it is a header-only,
# dependency-free, env-gated (NNTR_LAYER_PROFILE) scope timer that the CausalLM
# application's layers include. An application built out of tree sees only the
# INSTALLED header directory (include/nntrainer), so leaving this one
# uninstalled breaks that build with "'layer_prof.h' file not found". Installed,
# therefore, and listed in packaging/nntrainer.spec and
# debian/nntrainer-dev.install accordingly.
nntrainer_headers += meson.current_source_dir() / 'layer_prof.h'

# fused_llm_properties.h and nntr_htp_decode_hook.h are not layers either: the
# first holds the property types the CausalLM layers share, the second the
# HTP decode hook interface. CausalLM's causallm_common_properties.h, mha_core.h
# and htp_decode_hook.h include them, so an out-of-tree build (Android's
# android_build_result/include/nntrainer) fails without them. Installed for the
# same reason as layer_prof.h, and listed in packaging/nntrainer.spec and
# debian/nntrainer-dev.install.
nntrainer_headers += meson.current_source_dir() / 'fused_llm_properties.h'
nntrainer_headers += meson.current_source_dir() / 'nntr_htp_decode_hook.h'

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
