#ifndef HAP_PERF_H
#define HAP_PERF_H
#include <stdint.h>
static inline uint64_t HAP_perf_get_qtimer_count(void){return 0;}
static inline uint64_t HAP_perf_qtimer_count_to_us(uint64_t c){return c;}
#endif
