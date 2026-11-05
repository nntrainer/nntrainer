#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check shared WH records against pre-refactor bytes and both reader paths."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from htp_weight_utils import Checkpoint, bf16_round, bf16_rows, write_affine_wh

fixtures = (
    (32, 32, '0512636eee27644971fc9767c1ed8225981cead194ee9631a1921d708ab76308'),
    (64, 288, 'f19383df4ffdc838c1d8462f958f944ba57cb6be4512bfcfa27d019b4ff95ea4'),
    (128, 512, '42892fdfcfd0df5dde7212294011e937d8d4c1c740cb2d118ee3baa243fb8074'),
)
for k, n, digest in fixtures:
    codes = (np.arange(k*n).reshape(n, k) % 16 - 8).astype(np.int8)
    scales = (np.arange(n) % 7 + 1).astype(np.float32) / 128
    offsets = (np.arange(n) % 3 - 1).astype(np.int32)
    weights = bf16_round((codes + offsets[:, None]) * scales[:, None])
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw = (weights.view(np.uint32) >> 16).astype('<u2').tobytes()
        header = json.dumps({'weight': dict(dtype='BF16', shape=[n, k],
                                           data_offsets=[0, len(raw)])}).encode()
        path = directory / 'model.safetensors'
        path.write_bytes(struct.pack('<Q', len(header)) + header + raw)
        checkpoint = Checkpoint(directory)
        try:
            assert np.array_equal(checkpoint.read('weight'), weights)
            for row0, values in bf16_rows(path, 'weight', [n, k]):
                assert np.array_equal(values, checkpoint.read('weight', row0, row0 + len(values)))
            item = dict(name='fixture', shape=[k, n], offset=13)
            output = io.BytesIO()
            write_affine_wh(output, item, bf16_rows(path, 'weight', [n, k]))
            assert hashlib.sha256(output.getvalue()).hexdigest() == digest
        finally:
            checkpoint.close()
print('HTP converter: checkpoint rows and 3 pre-refactor WH byte fixtures PASS')
