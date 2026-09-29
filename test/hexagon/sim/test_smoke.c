// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_smoke.c
 * @date	18 August 2026
 * @brief	Hexagon-sim smoke test covering QuRT threads, HVX and VTCM acquire
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <qurt.h>
#include <stdio.h>
#include <string.h>

static void thread_fn(void *arg) { *(volatile int *)arg = 1; }

int test_smoke(void) {
  int units = qurt_hvx_get_units();
  printf("SIM_TEST smoke hvx_units=%d\n", units);
  if (units <= 0)
    return 1;

  /* QuRT thread round-trip */
  static char stack[8192] __attribute__((aligned(16)));
  volatile int flag = 0;
  qurt_thread_t tid;
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, sizeof(stack));
  qurt_thread_attr_set_priority(&attr, 100);
  if (qurt_thread_create(&tid, &attr, thread_fn, (void *)&flag) != QURT_EOK)
    return 1;
  int status;
  qurt_thread_join(tid, &status);
  if (flag != 1)
    return 1;

  /* HVX vadd: 128B int8 */
  if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK)
    return 1;
  static int8_t a[128] __attribute__((aligned(128)));
  static int8_t b[128] __attribute__((aligned(128)));
  static int8_t y[128] __attribute__((aligned(128)));
  for (int i = 0; i < 128; ++i) {
    a[i] = (int8_t)i;
    b[i] = 3;
  }
  *(HVX_Vector *)y =
    Q6_Vb_vadd_VbVb(*(const HVX_Vector *)a, *(const HVX_Vector *)b);
  qurt_hvx_unlock();
  for (int i = 0; i < 128; ++i)
    if (y[i] != (int8_t)(i + 3))
      return 1;

  /* VTCM 1MB acquire/release */
  compute_res_attr_t rattr;
  HAP_compute_res_attr_init(&rattr);
  HAP_compute_res_attr_set_vtcm_param(&rattr, 1024 * 1024, 1);
  unsigned ctx_id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
  if (ctx_id == 0) {
    printf("SIM_TEST smoke vtcm acquire fail\n");
    return 1;
  }
  void *vtcm = HAP_compute_res_attr_get_vtcm_ptr(&rattr);
  if (!vtcm) {
    HAP_compute_res_release(ctx_id);
    return 1;
  }
  memset(vtcm, 0xA5, 1024 * 1024);
  HAP_compute_res_release(ctx_id);

  printf("SIM_TEST smoke PASS\n");
  return 0;
}
