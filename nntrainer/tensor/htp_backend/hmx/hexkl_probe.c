// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_probe.c
 * @date 08 Aug 2026
 * @brief Five counters that split layer_run's time (Tier 0 measurement)
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items */

#include "hexkl_probe.h"

#include <string.h>

uint64_t hexkl_probe_us[HEXKL_PROBE_N];
int hexkl_probe_on;

void hexkl_probe_reset(int enable) {
  memset(hexkl_probe_us, 0, sizeof(hexkl_probe_us));
  hexkl_probe_on = enable;
}
