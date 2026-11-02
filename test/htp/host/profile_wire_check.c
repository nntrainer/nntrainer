// SPDX-License-Identifier: Apache-2.0
/**
 * @file profile_wire_check.c
 * @brief Verify the existing profiling RPC wire ordinals without the SDK.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include <htp_profile_wire.h>
#if defined(__cplusplus)
#define CHECK static_assert
#else
#define CHECK _Static_assert
#endif
CHECK(HTP_N_STAGES == 7 && HTP_T_ACC_STRIDE == 6, "FC wire layout");
CHECK(HTP_FU_N_STAGES == 8 && HTP_FU_T_ACC_COPY == 5, "fused wire layout");
CHECK(HTP_GU_N_STAGES == 7 && HTP_GU_T_DRAIN == 5, "gate/up wire layout");
CHECK(HTP_MOE_N_STAGES == 31 && HTP_MOE_T_PATH == 29 &&
        HTP_MOE_T_M1_FEED == 30 && HTP_MOE_T_DMA_DESC == 19,
      "MoE wire layout");
int main(void) { return 0; }
