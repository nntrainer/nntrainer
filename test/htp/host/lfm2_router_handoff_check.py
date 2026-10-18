#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check actual ARM staging and DSP RPC/carve helpers without DSP arithmetic."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cxx', default='c++')
p.add_argument('--mutant', choices=['stage', 'copy', 'reserve', 'take', 'ticket', 'prefix', 'restore', 'source_alias', 'sync', 'attn_q', 'attn_f16', 'id', 'arm_source', 'scratch_take', 'scratch_ticket', 'scratch_epoch', 'scratch_owner', 'scratch_bytes'])
a = p.parse_args()
a.repo = a.repo.resolve()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
paths = {
 'arm': 'nntrainer/tensor/htp_backend/htp_compute_ops.cpp',
 'dsp': 'test/htp/nntr_hvx_mm_u8i4.c',
 'core': 'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c',
 'scratch': 'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.h',
 'state': 'nntrainer/tensor/htp_backend/htp_router_handoff.h',
}
sources = {k: (a.repo / v).read_text() for k, v in paths.items()}
def block(s, begin, end):
    at = s.index(begin)
    return s[at:s.index(end, at)]

def body(s, signature):
    at = s.index(signature)
    start = s.index('{', at)
    depth = 1
    i = start + 1
    while depth:
        depth += (s[i] == '{') - (s[i] == '}')
        i += 1
    return s[at:i] + '\n'

parts = {}
parts['state.h'] = sources['state']
parts['pool.inc'] = block(sources['arm'], '  struct Held {', '  /** @brief A host range last copied')
parts['producer.inc'] = block(sources['arm'], '    const uint32_t ticket =', '    const uint64_t t0 =')
parts['ion_alias.inc'] = body(sources['arm'], '  const void *ionAlias(')
parts['sync.inc'] = body(sources['arm'], '  void sync_to_host(').replace(' override', '')
parts['attn_q.inc'] = body(sources['arm'], '  bool sdpa_q_kvcache(').replace(' override', '')
parts['attn_f16.inc'] = body(sources['arm'], '  bool sdpa_fp16_kvcache(').replace(' override', '')
parts['take_id.inc'] = body(sources['arm'], '  bool take_decode_token_id(').replace(' override', '')
# Check before-write anchoring in the large queue/materialization methods.
writer_anchors = {
    'release_prefill_scratch': ('  void release_prefill_scratch()', 'stagedMemcpy('),
    'tokenForward': ('  void tokenForward(', 'std::memcpy(q.act->data(),'),
    'oprojection_materialize': ('    if (u16_in && u16_off) {', 'float *dst ='),
}
for name, (begin, write) in writer_anchors.items():
    section = block(sources['arm'], begin, write)
    start = section.index('    std::lock_guard<std::mutex> lock(invoke_mutex_);')
    guard = '    htp_router_handoff_invalidate(&router_handoff_);'
    assert guard in section[start:], name
    parts[name + '.inc'] = section[start:section.index(guard, start) + len(guard)]
parts['select.inc'] = block(sources['arm'], '    HtpRpcBuffer &act_stage =\n      via_dspq', '    // NNTR_HTP_MOE_DEBUG=1')
parts['wire.inc'] = block(sources['arm'], '                        handoff_plain && rep == 0', ',\n                        h_gu.data()').strip()
for method in ['beginPrompt', 'releasePrefillStaging']:
    s = body(sources['arm'], '  void ' + method + '()')
    assert s[s.index('{') + 1:].lstrip().startswith('htp_router_handoff_invalidate(&router_handoff_);')
    parts[method + '.inc'] = s[s.index('{') + 1:s.index(';') + 1]
h = sources['scratch']
end = h.index('} hexkl_moe_scratch;') + len('} hexkl_moe_scratch;')
rotation_end = h.index('} hexkl_moe_rotated_input;') + len('} hexkl_moe_rotated_input;')
parts['scratch.inc'] = h[h.rindex('typedef struct {', 0, rotation_end):rotation_end] + '\n'
parts['scratch.inc'] += h[h.rindex('typedef struct {', 0, end):end]
parts['had_flags.inc'] = block(h, '#define HEXKL_MOE_FLAG_HAD_UP', '/** @brief Every bit')
parts['reserve.inc'] = block(sources['core'], 'void hexkl_moe_scratch_free(', '/** @brief Kept in sync')
parts['rpc.inc'] = block(sources['dsp'], 'static int check_moe_layer_args(', 'int nntr_hvx_router_logits_u8i8_f32(')
parts['core_enter.inc'] = block(sources['core'], '  const float *router_act = scratch ?', '  if (!tbl || !vtcm_base')
parts['core_layout.inc'] = block(sources['core'], '  const size_t n_slots_cap =', '  moe_m1_ctx m1;')
parts['core_slots.inc'] = block(sources['core'], '  {\n    uint32_t base = 0u, slot = 0u;', '  HEXKL_PROBE_T0(p0);')
parts['core_copy.inc'] = block(sources['core'], '  if (router_reuse) {', '  /** The HMX path does not clear out:')
# Keep the native compact-handle predicate anchored to actual validated handles.
assert 'n_compact_tl2 += g->tl2_compact && d->tl2_compact;' in sources['core']
assert 'n_compact_tl2 == n_with_rows' in parts['core_layout.inc']
if a.mutant:
    key, old, new = {
        'stage': ('pool.inc', '    // Handing out a writable buffer ends any pending router ownership.\n    htp_router_handoff_invalidate(&router_handoff_);', '    // mutation: missed writable owner'),
        'copy': ('pool.inc', '    // No-write hits above keep their ION epoch. Copies/swaps below do not.\n    htp_router_handoff_invalidate(&router_handoff_);', '    // mutation: missed input copy'),
        'reserve': ('reserve.inc', 'int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes) {\n  htp_router_handoff_scratch_invalidate(&s->router_handoff);', 'int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes) {'),
        'take': ('state.h', '  htp_router_handoff_invalidate(s);\n  return hit;', '  return hit;'),
        'ticket': ('state.h', 's->ticket == ticket', 'ticket != 0u'),
        'arm_source': ('state.h', 's->source == source', 'source != NULL'),
        'scratch_take': ('state.h', '  htp_router_handoff_scratch_invalidate(s);\n  return hit;', '  return hit;'),
        'scratch_ticket': ('state.h', 's->owner == owner && s->bytes == bytes', 's->owner == owner && s->bytes == bytes'),
        'scratch_epoch': ('state.h', 's->owner == owner && s->bytes == bytes', 's->owner == owner && s->bytes == bytes'),
        'scratch_owner': ('state.h', 's->owner == owner && s->bytes == bytes', 'owner != NULL && s->bytes == bytes'),
        'scratch_bytes': ('state.h', 's->owner == owner && s->bytes == bytes', 's->owner == owner && bytes != 0u'),
        'prefix': ('core_layout.inc', 'float *act_c = router_reuse ? (float *)hexkl_moe_carve(&cur, sz_act_c) : NULL;', 'float *act_c = router_reuse ? (float *)cur : NULL;'),
        'attn_q': ('attn_q.inc', '    htp_router_handoff_invalidate(&router_handoff_);', '    /* mutation: direct output not invalidated */'),
        'attn_f16': ('attn_f16.inc', '    htp_router_handoff_invalidate(&router_handoff_);', '    /* mutation: direct output not invalidated */'),
        'id': ('take_id.inc', '    htp_router_handoff_invalidate(&router_handoff_);', '    /* mutation: public id output not invalidated */'),
        'sync': ('sync.inc', '      htp_router_handoff_invalidate(&router_handoff_);', '      /* mutation: missing external destination write */'),
        'source_alias': ('producer.inc', re.search(r'!htp_router_handoff_overlap\(x, x_len \* sizeof\(float\), act,\s*x_len \* sizeof\(float\)\)', parts['producer.inc']).group(0), 'true'),
        'restore': ('select.inc', re.search(r'!stagingRestoreOverlaps\(out_pool_, out_bytes, act_stage.data\(\),\s*act_bytes\)', parts['select.inc']).group(0), 'true'),
    }[a.mutant]
    assert old in parts[key], (a.mutant, 'source anchor changed')
    if a.mutant.startswith('scratch_'):
        at = parts[key].index('htp_router_handoff_scratch_take(')
        prefix, suffix = parts[key][:at], parts[key][at:]
        if a.mutant == 'scratch_ticket':
            old, new = 's->ticket == ticket', 'ticket != 0u'
        if a.mutant == 'scratch_epoch':
            old, new = 's->epoch == s->published_epoch', '1'
        assert old in suffix
        parts[key] = prefix + suffix.replace(old, new, 1)
    else:
        parts[key] = parts[key].replace(old, new, 1)
for name, text in parts.items():
    (a.output / name).write_text(text)
fixture = Path(__file__).with_suffix('.cpp')
shutil.copy2(fixture, a.output / fixture.name)
binpath = a.output / 'router_handoff_check'
cmd = [a.cxx, '-std=c++17', '-O1', '-g', '-Wall', '-Wextra',
       '-Wno-unused-variable', '-Wno-unused-parameter', '-Wno-unused-but-set-variable', '-pthread',
       '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
       str(a.output / fixture.name), '-o', str(binpath)]
manifest = {
    'source_sha256': {paths[k]: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
    'extracted_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in parts.items()},
    'mutant': a.mutant, 'compile': cmd, 'run': [str(binpath)],
    'limits': 'Real staging, ownership, RPC guards, reserve and carve helpers. Arithmetic, DMA and allocation are host stand-ins; no model-output or hardware-performance claim.',
}
(a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(cmd, check=True)
subprocess.run([str(binpath)], check=True)
