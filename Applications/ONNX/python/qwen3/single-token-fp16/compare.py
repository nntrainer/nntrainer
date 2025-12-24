"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>

@file compare.py
@date 24 December 2025
@brief This script compares official and NNTrainer model output logits.
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sumon Nath <sumon.nath@samsung.com>
"""

import numpy as np

arr1 = np.fromfile("./modelling_logits.bin", dtype="float16").reshape(1, 151936)
arr2 = np.fromfile("./nntrainer_logits.bin", dtype="float32").reshape(1, 151936)
print(arr1)
print(arr2)

if np.allclose(arr1, arr2, atol=1e-4, rtol=1e-4):
    print("equal")
else:
    print("not equal")
