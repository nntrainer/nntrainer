"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2026 Sumon Nath <sumon.nath@samsung.com>

@file onnx_exporter.py
@date 6 January 2026
@brief This script creates weight.bin for each weight of the ONNX model and converts all ONNX weights to FP32
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sumon Nath <sumon.nath@samsung.com>
"""

import onnx
import numpy as np
import json
from onnx import numpy_helper, TensorProto


def cleanName(name):
    if name.startswith("/"):
        name = name[1:]

    name = name.replace("/", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")
    name = name.lower()

    return name


model = onnx.load("./model.onnx", load_external_data=True)

metadata = {}


for tensor in model.graph.initializer:
    arr = numpy_helper.to_array(tensor)

    filename = f"./{cleanName(tensor.name)}.bin"
    arr.tofile(filename)

    # Save metadata (name, dtype, shape, file)
    metadata[tensor.name] = {
        "file": filename,
        "tensor name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": list(arr.shape),
    }

    print(f"Saved {tensor.name} -> {filename}, dtype={arr.dtype}, shape={arr.shape}")
