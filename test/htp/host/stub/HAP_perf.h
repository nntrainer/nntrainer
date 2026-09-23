#ifndef HAP_PERF_H
#define HAP_PERF_H
#include <stdint.h>
static inline uint64_t HAP_perf_get_qtimer_count(void){return 0;}
static inline uint64_t HAP_perf_qtimer_count_to_us(uint64_t c){return c;}
/* Monotonic and never equal twice, so a per-op pcycle bracket of an op
   that ran reads > 0 and one that did not reads 0 (graph_host_check). */
static uint64_t host_stub_pcycles;
static inline uint64_t HAP_perf_get_pcycles(void) {
  return ++host_stub_pcycles;
}
#endif
