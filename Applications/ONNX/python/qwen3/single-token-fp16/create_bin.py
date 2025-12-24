"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>

@file create_bin.py
@date 25 December 2025
@brief This script creates weight.bin for each weight of the ONNX model and converts all ONNX weights to FP32.
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sumon Nath <sumon.nath@samsung.com>
"""

import onnx
import numpy as np
import json
import os
import shutil
from onnx import numpy_helper, TensorProto


def cleanName(name):
    if name.startswith("/"):
        name = name[1:]

    name = name.replace("/", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")
    name = name.lower()

    return name


model_name = "./qwen3_model.onnx"
model = onnx.load(model_name, load_external_data=True)
new_initializers = []

metadata = {}

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "bins")
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

for tensor in model.graph.initializer:
    arr = numpy_helper.to_array(tensor).astype(np.float16)
    if tensor.data_type != TensorProto.FLOAT16:
        new_tensor = numpy_helper.from_array(arr, name=tensor.name)
        new_initializers.append(new_tensor)
    else:
        new_initializers.append(tensor)
    filename = f"./bins/{cleanName(tensor.name)}.bin"
    arr.tofile(filename)

    # Save metadata (name, dtype, shape, file)
    metadata[tensor.name] = {
        "file": filename,
        "tensor name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": list(arr.shape),
    }

    print(f"Saved {tensor.name} -> {filename}, dtype={arr.dtype}, shape={arr.shape}")

# Replace initializers
model.graph.initializer.clear()
model.graph.initializer.extend(new_initializers)
onnx.save(
    model,
    model_name,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    size_threshold=0,
)

with open("./weights_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
