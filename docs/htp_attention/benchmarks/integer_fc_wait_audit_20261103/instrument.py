from pathlib import Path
import shutil

original = Path('/home/jungone-lee/workspace/nntrainer-pr4471-integer-fc')
root = Path('/tmp/nntr-w4a16-waits-20261103')
src = root / 'source'
backend = src / 'nntrainer/tensor/htp_backend'
shutil.copytree(original / 'nntrainer/tensor/htp_backend', backend, dirs_exist_ok=True)
for p in (original / 'nntrainer/tensor').glob('*.h'):
    shutil.copy2(p, backend.parent / p.name)
test = src / 'test/htp'
test.mkdir(parents=True, exist_ok=True)
for p in (original / 'test/htp').iterdir():
    if p.is_file() and p.suffix in ('.c', '.h', '.idl', '.sh'):
        shutil.copy2(p, test / p.name)

names = ['setup', 'control', 'dma_push', 'pack_submit', 'dma_first', 'dma_next',
         'pack_first', 'pack_next', 'core', 'ep_wait', 'ep_final', 'status',
         'fence', 'ep_submit', 'cleanup', 'sweep', 'read_issue']
header = '''#ifndef FC_TRACE_H
#define FC_TRACE_H
#include <stdint.h>
#include <HAP_perf.h>
enum { ''' + ', '.join('FC_' + x.upper() for x in names) + ''', FC_COUNT };
extern int fc_trace_mode;
extern uint64_t fc_trace_ticks[FC_COUNT];
extern uint32_t fc_trace_counts[FC_COUNT], fc_trace_meta[8];
extern uint64_t fc_trace_last;
extern unsigned fc_trace_stage;
void fc_trace_reset(int mode);
static inline void fc_stage(unsigned s) {
  if (!fc_trace_mode) return;
  uint64_t now = HAP_perf_get_qtimer_count();
  if (fc_trace_last) fc_trace_ticks[fc_trace_stage] += now - fc_trace_last;
  fc_trace_last = now;
  fc_trace_stage = s;
  ++fc_trace_counts[s];
}
static inline void fc_finish(void) {
  if (!fc_trace_mode) return;
  fc_stage(FC_CONTROL);
  fc_trace_last = 0;
}
#endif
'''
(backend / 'hmx/fc_trace.h').write_text(header)
p = backend / 'hmx/hexkl_mm_u8i4_dma.c'
s = p.read_text().replace('#include "hexkl_probe.h"', '''#include "hexkl_probe.h"
#include "fc_trace.h"
int fc_trace_mode;
uint64_t fc_trace_ticks[FC_COUNT], fc_trace_last;
uint32_t fc_trace_counts[FC_COUNT], fc_trace_meta[8];
unsigned fc_trace_stage;
void fc_trace_reset(int mode) {
  memset(fc_trace_ticks, 0, sizeof fc_trace_ticks);
  memset(fc_trace_counts, 0, sizeof fc_trace_counts);
  fc_trace_last = 0;
  fc_trace_stage = FC_SETUP;
  fc_trace_mode = mode;
}
''')
start = s.index('static int fc_layer_run(')
stop = s.index('\nint hexkl_mm_u8i4_layer_run(', start)
part = s[start:stop]
def change(a, b, count=1):
    global part
    assert part.count(a) == count, (a, part.count(a), count)
    part = part.replace(a, b)
change('  static const hexkl_mm_opts kDefaults = {0};', '  fc_stage(FC_SETUP);\n  static const hexkl_mm_opts kDefaults = {0};')
change('  // Reset once per call,', '''  if (fc_trace_mode) {
    fc_trace_meta[0] = panel_handles;
    fc_trace_meta[1] = acc_tiles;
    fc_trace_meta[2] = n_rblocks;
    fc_trace_meta[3] = n_handles;
    fc_trace_meta[4] = 2u * act_bytes;
    fc_trace_meta[5] = weight_panel_bytes;
    fc_trace_meta[6] = result_off;
    fc_trace_meta[7] = arena;
  }
  // Reset once per call,''')
change('  for (uint32_t ph = 0; ph < panel_handles; ++ph) {', '  fc_stage(FC_DMA_PUSH);\n  for (uint32_t ph = 0; ph < panel_handles; ++ph) {')
change('    hvx_worker_pool_submit_bg(o->pool, &pack_job);', '    fc_stage(FC_PACK_SUBMIT);\n    hvx_worker_pool_submit_bg(o->pool, &pack_job);\n    fc_stage(FC_SETUP);')
change('  HEXKL_PROBE_T0(p0);\n  hexkl_dma_ring_drain();\n  HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);', '  fc_stage(FC_DMA_FIRST);\n  HEXKL_PROBE_T0(p0);\n  hexkl_dma_ring_drain();\n  HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);\n  fc_stage(FC_CONTROL);')
change('    for (uint32_t ph = 0; ph < panel_handles && next + ph < n_handles; ++ph) {', '    fc_stage(FC_DMA_PUSH);\n    for (uint32_t ph = 0; ph < panel_handles && next + ph < n_handles; ++ph) {')
change('    size_t panel_end = out_off;', '    fc_stage(FC_CONTROL);\n    size_t panel_end = out_off;')
change('        hvx_worker_pool_wait_bg(o->pool, &pack_job, (rb + 1u) * 16u);', '        fc_stage(rb ? FC_PACK_NEXT : FC_PACK_FIRST);\n        hvx_worker_pool_wait_bg(o->pool, &pack_job, (rb + 1u) * 16u);\n        fc_stage(FC_CONTROL);')
change('          HEXKL_PROBE_T0(mm_begin);', '          fc_stage(FC_CORE);\n          HEXKL_PROBE_T0(mm_begin);')
change('              const uint32_t plane = sw == 0u ? act_hi_off : act_off;', '              if (fc_trace_mode == 2) fc_stage(FC_SWEEP);\n              const uint32_t plane = sw == 0u ? act_hi_off : act_off;')
change('              HEXKL_PROBE_T0(p0);\n              const uint32_t slot_off =', '              if (fc_trace_mode == 2) fc_stage(FC_READ_ISSUE);\n              HEXKL_PROBE_T0(p0);\n              const uint32_t slot_off =')
change('          HEXKL_PROBE_ADD(HEXKL_PROBE_MM, mm_begin);', '          HEXKL_PROBE_ADD(HEXKL_PROBE_MM, mm_begin);\n          fc_stage(FC_CONTROL);')
change('              hvx_worker_pool_wait(o->pool);', '              fc_stage(FC_EP_WAIT);\n              hvx_worker_pool_wait(o->pool);\n              fc_stage(FC_STATUS);')
change('              integer_planes_fence(vtcm_base,', '              fc_stage(FC_FENCE);\n              integer_planes_fence(vtcm_base,')
change('              hvx_worker_pool_submit(o->pool, hexkl_i32_batch_worker, batch,', '              fc_stage(FC_EP_SUBMIT);\n              hvx_worker_pool_submit(o->pool, hexkl_i32_batch_worker, batch,')
change('              stage_used = 0;', '              fc_stage(FC_CONTROL);\n              stage_used = 0;')
change('    hexkl_dma_ring_drain();\n    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);', '    fc_stage(FC_DMA_NEXT);\n    hexkl_dma_ring_drain();\n    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);\n    fc_stage(FC_CONTROL);', 2)
change('\n  hvx_worker_pool_wait(o->pool);', '\n  fc_stage(FC_EP_FINAL);\n  hvx_worker_pool_wait(o->pool);\n  fc_stage(FC_CLEANUP);')
change('  return rc;\n}', '  fc_finish();\n  return rc;\n}')
s = s[:start] + part + s[stop:]
p.write_text(s)

p = test / 'integer_fc_device_bench.c'
s = p.read_text()
s = '#include "fc_trace.h"\n' + s
a = s.index('  hexkl_probe_reset(1);')
b = s.index('  const uint32_t rs[3]', a)
profile = '''  hexkl_probe_on = 0;
  for (unsigned pass = 0; pass < 8; ++pass) {
    REQUIRE(hexkl_mm_u16i4_layer_run_i32_requant(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
      handles, nh, a, 30000, NULL, y, s->quant_pool, rpp, nr, q) == 0);
    fc_trace_reset(pass < 5 ? 1 : 2);
    uint64_t profile_start = HAP_perf_get_qtimer_count();
    REQUIRE(hexkl_mm_u16i4_layer_run_i32_requant(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
      handles, nh, a, 30000, NULL, y, s->quant_pool, rpp, nr, q) == 0);
    double profile_us = us(profile_start);
    fc_trace_mode = 0;
    printf("TRACE M=%u K=%u N=%u pass=%u mode=%u total_us=%.3f", M, K, N, pass,
      pass < 5 ? 1u : 2u, profile_us);
'''
for i, n in enumerate(names):
    profile += f'    printf(" {n}_us=%.3f {n}_n=%u", fc_trace_ticks[{i}]/19.2, fc_trace_counts[{i}]);\n'
profile += '''    printf(" panel_handles=%u acc_tiles=%u row_blocks=%u handles=%u act_bytes=%u panel_bytes=%u result_off=%u arena=%u\\n",
      fc_trace_meta[0], fc_trace_meta[1], fc_trace_meta[2], fc_trace_meta[3],
      fc_trace_meta[4], fc_trace_meta[5], fc_trace_meta[6], fc_trace_meta[7]);
  }
'''
s = s[:a] + profile + s[b:]
p.write_text(s)
build = (Path('/tmp/nntr-integer-bench/build_bench.sh')).read_text()
build = build.replace('repo=/home/jungone-lee/workspace/nntrainer-pr4471-integer-fc', f'repo={src}')
build = build.replace('out=/tmp/nntr-integer-bench/$variant', f'out={root}/binary')
(root / 'build_bench.sh').write_text(build)
print(src)
