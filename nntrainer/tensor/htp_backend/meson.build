htp_backend_sources = [
  'htp_backend.cpp',
  'htp_compute_ops.cpp',
]

htp_backend_headers = [
  'htp_backend.h',
  'htp_rpcmem.h',
  'htp_graph_desc.h',
]

foreach s : htp_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : htp_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

# FastRPC client stub for the same IDL the DSP skel (test/htp/build.sh) is
# built from. Not a custom_target: qaic is a proprietary Hexagon SDK tool a
# normal build host does not have, and test/htp/build.sh already established
# the convention of a manual generation step for it. generate_stub.sh puts
# its output where this checks for it; both nntrainer_sources (this build)
# and jni/meson.build's ndk-build source list (the Android build) read
# nntrainer_sources by absolute path, so one entry here reaches both.
nntr_hvx_stub_c = meson.current_source_dir() / 'generated' / 'nntr_hvx_stub.c'
nntr_hvx_idl = meson.project_source_root() / 'test' / 'htp' / 'nntr_hvx.idl'
if not fs.is_file(nntr_hvx_stub_c)
  error('enable-htp=true requires the generated FastRPC stub. Run:\n' +
        '  HEXAGON_SDK_ROOT=<path> ' +
        meson.current_source_dir() / 'generate_stub.sh')
endif
# A MISSING stub is caught above; a STALE one used to sail through and fail
# later as an undeclared identifier in htp_compute_ops.cpp, which names the
# symbol but not the reason. Worse, test/htp/build.sh regenerates a SECOND
# copy under test/htp/generated for the DSP skel, so running only that one
# leaves a fresh skel against a stale client -- the device then rejects the
# call with EBADPARM and the build looks fine. Compare against the IDL both
# are generated from and say so here instead.
if fs.is_file(nntr_hvx_idl)
  stub_h = meson.current_source_dir() / 'generated' / 'nntr_hvx.h'
  if fs.is_file(stub_h)
    idl_newer = run_command(
      'test', nntr_hvx_idl, '-nt', stub_h, check: false).returncode() == 0
    if idl_newer
      error('The generated FastRPC stub is older than ' + nntr_hvx_idl +
            '.\nRegenerate the ARM client stub (test/htp/build.sh only does ' +
            'the DSP skel):\n' +
            '  HEXAGON_SDK_ROOT=<path> ' +
            meson.current_source_dir() / 'generate_stub.sh')
    endif
  endif
endif
if get_option('htp-inproc')
  # [#84] The skel compiled into the process: every nntr_hvx_* entry the
  # ARM side calls is a direct call into the same C function the DSP runs,
  # on scalar stand-ins for the HMX unit (standin/hvx_scalar.c), the
  # FastRPC runtime (inproc/rpc_standin.c), HexKL's micro API
  # (inproc/hexkl_micro_standin.c), QuRT (stub/qurt.h, pthreads) and the
  # HVX f32 intrinsics (hvx_emu/). The qaic header above is still the one
  # the entries are declared in, so the prototypes are the IDL's.
  #
  # What is in and what is out: the entries the ARM side references
  # (open/close, the u8i4 weight and MoE calls, the arena, moe_set_opts,
  # the #85 graph entries) plus the #82/#81 small-op and m=1 attention
  # entries, whose kernels already compile on hvx_emu, and the #141 MoE
  # dspqueue thread against inproc/dspqueue_standin.c. Left out, because
  # nothing here references them and their kernels need intrinsics
  # hvx_emu does not emulate: nntr_hvx_softmax.c, nntr_hvx_attn.c,
  # nntr_hvx_mm_u8i8.c, nntr_hvx_dma_probe.c (device-gtest-only entries).
  # A skel entry the ARM calls but this list lacks is a link error
  # (--no-undefined below), the host twin of test/htp/build.sh's guard.
  htp_test_dir = meson.source_root() / 'test' / 'htp'
  htp_host_dir = htp_test_dir / 'host'
  inproc_sources = files(
    htp_test_dir / 'hvx_add_f32.c',
    htp_test_dir / 'nntr_hvx_mm_u8i4.c',
    htp_test_dir / 'nntr_hvx_graph.c',
    htp_test_dir / 'nntr_hvx_small_ops.c',
    htp_test_dir / 'nntr_hvx_attn_m1.c',
    htp_test_dir / 'nntr_hvx_dspq.c',
    htp_test_dir / 'nntr_hvx_token.c',
    htp_test_dir / 'nntr_hvx_fc_q4.c',
    htp_test_dir / 'nntr_hvx_sf_probe.c',
    'hmx/hexkl_mm_u8i4.c',
    'hmx/hexkl_mm_u8i4_dma.c',
    'hmx/hexkl_mm_u8i4_moe.c',
    'hvx/hvx_hadamard_f32.c',
    'hmx/hexkl_mm_u8i8_dma.c',
    'hmx/hexkl_conv_block.c',
    'hmx/hexkl_graph.c',
    'hmx/hexkl_token.c',
    'hmx/hexkl_dma_ring.c',
    'hmx/hexkl_dma_trace.c',
    'hmx/hexkl_probe.c',
    'hmx/hexkl_acc_tile.c',
    # the session's close() releases the two KV tables of nntrainer#4343;
    # the attention entries that fill them are device-only (see above)
    'hmx/hexkl_kv_tiles_f16.c',
    # hmx/hexkl_kv_q.c: in nntrainer_sources already (cpu_backend, #4415)
    'hvx/hvx_worker_pool.c',
    'hvx/hvx_expand_i2i4.c',
    'hvx/hvx_gemv_u8tl2.c',
    'hvx/hvx_gather_ah_u8.c',
    'hvx/hvx_scale_add_f32.c',
    'hvx/hvx_conv_gate_f32.c',
    'hvx/hvx_m1_ops_f32.c',
    'hvx/hvx_attn_m1_f32.c',
    'hvx/hvx_q4_gemv_f32.c',
    # [#260] #4415's fused prefill entries (norm / RoPE / router rows,
    # the lm_head softcap)
    'hvx/hvx_rmsnorm_rows_f32.c',
    'hvx/hvx_rope_rows_f32.c',
    'hvx/hvx_router_rows_f32.c',
    'hvx/hvx_softmax_f32.c',
    'hvx/hvx_softcap_f32.c',
    htp_host_dir / 'standin' / 'hvx_scalar.c',
    htp_host_dir / 'inproc' / 'hexkl_micro_standin.c',
    htp_host_dir / 'inproc' / 'rpc_standin.c',
    htp_host_dir / 'inproc' / 'dspqueue_standin.c',
  )
  # inproc/ first: it shadows HAP_perf.h (the SDK's reads Hexagon
  # registers) and provides hexagon_protos.h; stub/ is reached through
  # inproc/'s forwarding headers so its AEEStdErr.h and HAP_perf.h do not
  # shadow the SDK's. -ffp-contract=off: the stand-ins and the emulated
  # kernels round every f32 op on its own, as the device does.
  # -include malloc.h: the m=1 attention cache is memalign(128), which the
  # Hexagon libc declares in stdlib.h and glibc in malloc.h.
  # -DNNTR_Q4M1_HOST_ID: the Q4M1 quantizer's x86-order switch
  # (NNTR_INPROC_X86_Q8, hvx_q4_gemv_f32.c; #132 Part B E1).
  nntr_hvx_inproc = static_library('nntr_hvx_inproc',
    inproc_sources,
    include_directories: include_directories(
      '../../../test/htp/host/inproc', '../../../test/htp/host/hvx_emu',
      '../../../test/htp/host/standin', '../../../test/htp',
      'generated', 'hmx', 'hvx', '.', '..'),
    dependencies: [hexkl_dep, thread_dep, libm_dep],
    c_args: ['-ffp-contract=off', '-include', 'malloc.h',
             '-Wno-format-truncation', '-DNNTR_Q4M1_HOST_ID',
             # hvx_softmax_util.h's float <-> bits puns (#4415's router)
             '-fno-strict-aliasing'],
    pic: true,
    build_by_default: false,
  )
  nntrainer_link_whole += nntr_hvx_inproc
  nntrainer_link_args += '-Wl,--no-undefined'
else
  nntrainer_sources += nntr_hvx_stub_c
endif
nntrainer_inc_abs += meson.current_source_dir() / 'generated'
nntrainer_inc += include_directories('generated')
