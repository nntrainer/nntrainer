// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	simple_test.cpp
 * @date	24 March 2026
 * @brief	Simple link test for flash attention OpenCL
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Pallavi Ravishankar <pallavi.r@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <flash_attention.h>
#include <iostream>
#include <cl_context.h>

int main() {
    std::cout << "Test program" << std::endl;
    
    // Try to call the function to see if it links
    // This is just a test call, not a real test
    _FP16* dummy = nullptr;
    nntrainer::flash_attention_fp16_cl(dummy, dummy, dummy, dummy, 0, 0, 0, 0, 0, 0, 0.0f);
    
    return 0;
}
