"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>

@file onnx_exporter.py
@date 25 December 2025
@brief This file inferences qwen model using custom_qwen3.py file and generates its ONNX model.
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sumon Nath <sumon.nath@samsung.com>
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_qwen3 import NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import torch
import numpy as np

model_name = "Qwen/Qwen3-1.7B"

official_model = AutoModelForCausalLM.from_pretrained(model_name)

qwenConfig = official_model.config
custom_model = NNTrainerQwen3ForCausalLM(qwenConfig)
custom_model.load_state_dict(official_model.state_dict(), strict=False)
# Convert to FP16 immediately after loading
custom_model = custom_model.half()
print("<Model converted to fp16>")

# Force all model parameters to FP16 to ensure no FP32 remnants
for param in custom_model.parameters():
    param.data = param.data.half()
print("<All model parameters forced to FP16>")

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

x = torch.tensor(
    [
        [
            52,
        ],
    ]
).view(-1, 1)
position_ids = torch.arange(1).reshape(1, -1)
cos, sin = rotary_emb(x, position_ids)
# Convert to tensors and immediately to FP16 (except position_ids which is used for indexing)
cos = torch.tensor(cos.numpy()).half()
sin = torch.tensor(sin.numpy()).half()
variance_epsilon = torch.tensor(
    [
        [
            1e-6,
        ]
    ]
).half()

# Ensure all inputs are on the same device as the model first
device = next(custom_model.parameters()).device
x = x.to(device)
cos = cos.to(device)
sin = sin.to(device)
variance_epsilon = variance_epsilon.to(device)

# Convert inputs to fp16 for consistency (except input_ids which must remain integer)
cos = cos.half()
sin = sin.half()
variance_epsilon = variance_epsilon.half()
# Note: x (input_ids) remains as integer type for gather operation

logits_of_custom_model = custom_model(x, cos, sin, variance_epsilon)
logits_of_official_model = official_model(x).logits

print("Logits of custom model: ")
print(logits_of_custom_model)
print("Logits of official model: ")
print(logits_of_official_model)

if (torch.allclose(logits_of_custom_model.float(), logits_of_official_model,atol=1e-3)):
    print("<All logits matched successfully>")
else:
    print("<Some logits do not match>")

logits_of_custom_model = logits_of_custom_model.detach().numpy()
logits_of_custom_model.tofile("./modelling_logits.bin")

torch.onnx.export(
    custom_model,
    (x, cos, sin, variance_epsilon),
    "qwen3_model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input", "cos", "sin", "variance_epsilon"],
    output_names=["output"],
    keep_initializers_as_inputs=False,
    dynamic_axes=None,
    do_constant_folding=True,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    training=torch.onnx.TrainingMode.EVAL,
)

print("<FP16 Model exported successfully>")
