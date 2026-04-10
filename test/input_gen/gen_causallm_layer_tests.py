#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate golden test data for CausalLM custom layers.

Golden file format (binary, sequential):
  For each tensor: [size_as_int32] [data_as_float32_array]

Order:
  1. Initial weights
  2. Inputs
  3. Outputs (after forward)
  4. Gradients (for trainable weights only)
  5. Weights (after backward - same as initial if frozen)
  6. Derivatives (outgoing derivatives / input gradients)

Note: incoming derivative is set to 2.0 in the test framework.
"""

import numpy as np
import os

# Fix seed for reproducibility (matches recorder.py SEED=1234)
SEED = 1234
np.random.seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_tensor(f, tensor):
    """Write a tensor in nnlayergolden format: [size_int32][data_float32]"""
    tensor = tensor.astype(np.float32)
    np.array(tensor.size, dtype=np.int32).tofile(f)
    tensor.tofile(f)


def rand_input(shape):
    """Generate random integer input [0, 10), matching recorder.py _rand_like"""
    return np.random.randint(0, 10, shape).astype(np.float32)


def gen_rms_norm_golden(input_shape=(2, 3, 3, 3), epsilon=1e-3,
                        filename="causallm_rmsnorm.nnlayergolden"):
    """Generate golden data for CausalLM RMS Norm layer with backward pass.

    RMS Norm forward: y = gamma * x / rms
    where rms = sqrt(mean(x^2, axis=-1, keepdim=True) + epsilon)

    Backward (incoming_deriv = 2.0):
    gamma_dy = gamma * incoming_deriv
    c = mean(gamma_dy * x, axis=-1, keepdim=True)
    dx = inv_rms * (gamma_dy - x * c * inv_rms^2)
    """
    width = input_shape[-1]

    # Initial weights: gamma = ones
    gamma = np.ones((1, 1, 1, width), dtype=np.float32)

    # Input
    x = rand_input(input_shape)

    # Forward pass
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)  # mean(x^2)
    inv_rms = 1.0 / np.sqrt(mean_sq + epsilon)
    output = x * inv_rms * gamma

    # Backward pass
    # incoming derivative = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # gamma_dy = gamma * incoming_deriv
    gamma_dy = gamma * incoming_deriv
    # c = mean(gamma_dy * x, axis=-1, keepdims=True)
    c = np.mean(gamma_dy * x, axis=-1, keepdims=True)
    # dx = inv_rms * (gamma_dy - x * c * inv_rms^2)
    dx = inv_rms * (gamma_dy - x * c * inv_rms * inv_rms)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights
        write_tensor(f, gamma)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (gamma is NOT trainable -> nothing to write)
        # 5. Weights (unchanged)
        write_tensor(f, gamma)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, gamma shape: {gamma.shape}")
    print(f"  output sample: {output.flat[:5]}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


def gen_lm_head_golden(input_shape=(2, 1, 1, 10), unit=5,
                       filename="causallm_lmhead.nnlayergolden"):
    """Generate golden data for CausalLM LM Head layer with backward pass.

    LM Head forward: output = input @ weight (+ bias)
    Weight shape: (1, 1, in_width, unit) for NCHW

    Backward (incoming_deriv = 2.0):
    dx = dy @ W^T
    """
    in_width = input_shape[-1]

    # Weight: shape (1, 1, in_width, unit)
    weight = rand_input((1, 1, in_width, unit))
    # Bias: disabled for this test

    # Input
    x = rand_input(input_shape)

    # Forward: output = x @ weight
    # x shape: (batch, 1, 1, in_width), weight: (1, 1, in_width, unit)
    # dot: (batch, 1, 1, in_width) x (1, 1, in_width, unit) -> (batch, 1, 1, unit)
    x_2d = x.reshape(-1, in_width)
    w_2d = weight.reshape(in_width, unit)
    out_2d = x_2d @ w_2d
    output = out_2d.reshape(input_shape[0], 1, 1, unit)

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # dx = dy @ W^T
    dy_2d = incoming_deriv.reshape(-1, unit)
    dx_2d = dy_2d @ w_2d.T
    dx = dx_2d.reshape(input_shape)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights (weight is trainable by default in LmHead)
        write_tensor(f, weight)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (weight gradient: dW = x^T @ dy)
        dw_2d = x_2d.T @ dy_2d
        dw = dw_2d.reshape(1, 1, in_width, unit)
        write_tensor(f, dw)
        # 5. Weights (unchanged - no optimizer step)
        write_tensor(f, weight)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, weight shape: {weight.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


def gen_swiglu_golden(input_shape=(2, 1, 1, 10),
                      filename="causallm_swiglu.nnlayergolden"):
    """Generate golden data for CausalLM SwiGLU layer with backward pass.

    SwiGLU forward: output = swish(gate) * up
    where swish(x) = x * sigmoid(x)

    Two inputs: gate (input_idx=0), up (input_idx=1)

    Backward (incoming_deriv = 2.0):
    d_up = swish(gate) * dy
    d_gate = up * swish'(gate) * dy
    where swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """
    # Inputs (no weights for SwiGLU)
    gate = rand_input(input_shape)
    up = rand_input(input_shape)

    # Forward
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    swish_gate = gate * sigmoid_gate
    output = swish_gate * up

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)

    # d_up = swish(gate) * dy
    d_up = swish_gate * incoming_deriv

    # swish'(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
    swish_prime = sigmoid_gate + gate * sigmoid_gate * (1.0 - sigmoid_gate)
    # d_gate = up * swish'(gate) * dy
    d_gate = up * swish_prime * incoming_deriv

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights (none)
        # 2. Inputs (gate, up)
        write_tensor(f, gate)
        write_tensor(f, up)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (no trainable weights)
        # 5. Weights (none)
        # 6. Derivatives (d_gate, d_up)
        write_tensor(f, d_gate)
        write_tensor(f, d_up)

    print(f"Generated: {filepath}")
    print(f"  gate shape: {gate.shape}, up shape: {up.shape}")
    print(f"  output sample: {output.flat[:5]}")
    print(f"  d_gate sample: {d_gate.flat[:5]}")
    print(f"  d_up sample: {d_up.flat[:5]}")

    return filepath


def gen_tie_word_embedding_golden(input_shape=(2, 1, 1, 10), unit=5,
                                  filename="causallm_tiewordembedding.nnlayergolden"):
    """Generate golden data for CausalLM TieWordEmbedding layer (lm_head mode).

    TieWordEmbedding in lm_head mode: output = input @ weight^T
    Weight shape: (1, 1, unit, in_width) — stored transposed vs LmHead.

    Backward (incoming_deriv = 2.0):
    dx = dy @ weight  (since forward was input @ weight^T)
    """
    in_width = input_shape[-1]

    # Weight: shape (1, 1, unit, in_width) — note: transposed vs LmHead
    weight = rand_input((1, 1, unit, in_width))

    # Input
    x = rand_input(input_shape)

    # Forward: output = x @ weight^T
    x_2d = x.reshape(-1, in_width)
    w_2d = weight.reshape(unit, in_width)  # (unit, in_width)
    out_2d = x_2d @ w_2d.T  # (batch, unit)
    output = out_2d.reshape(input_shape[0], 1, 1, unit)

    # Backward: incoming_deriv = 2.0
    incoming_deriv = np.full_like(output, 2.0)
    # dx = dy @ weight (not transposed, since forward used weight^T)
    dy_2d = incoming_deriv.reshape(-1, unit)
    dx_2d = dy_2d @ w_2d  # (batch, in_width)
    dx = dx_2d.reshape(input_shape)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights
        write_tensor(f, weight)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs
        write_tensor(f, output)
        # 4. Gradients (weight gradient: dW = (x^T @ dy)^T = dy^T @ x for transposed weight)
        dw_2d = dy_2d.T @ x_2d  # (unit, in_width)
        dw = dw_2d.reshape(1, 1, unit, in_width)
        write_tensor(f, dw)
        # 5. Weights (unchanged)
        write_tensor(f, weight)
        # 6. Derivatives
        write_tensor(f, dx)

    print(f"Generated: {filepath}")
    print(f"  input shape: {x.shape}, weight shape: {weight.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  derivative sample: {dx.flat[:5]}")

    return filepath


def gen_mha_core_golden(batch=2, seq_len=4, num_heads_q=4, num_heads_kv=2,
                        head_dim=8, is_causal=True, rope_theta=10000.0,
                        filename="causallm_mhacore.nnlayergolden"):
    """Generate golden data for CausalLM MHA Core layer.

    MHA Core: multi-head attention with RoPE and GQA support.
    3 inputs: Q (B,1,S,H_Q*D), K (B,1,S,H_KV*D), V (B,1,S,H_KV*D)
    1 output: (B,1,S,H_Q*D)

    No trainable weights in MHA Core (Q/K/V/O projections are separate FC layers).

    Note: RoPE thetas/cos/sin are computed in float32 matching C++ exactly.
    All attention computations use float64 for precision, casting to float32
    only at the final output to match C++ float32 results within tolerance.
    """
    gqa_size = num_heads_q // num_heads_kv
    q_width = num_heads_q * head_dim
    kv_width = num_heads_kv * head_dim
    half_dim = head_dim // 2

    # Compute RoPE frequencies matching C++ _compute_default_parameters exactly
    # C++: thetas[i] = float(1.0 / pow(theta, 2*i / float(head_dim)))
    thetas = np.zeros(half_dim, dtype=np.float32)
    for i in range(half_dim):
        exponent = np.float32(np.float32(2 * i) / np.float32(head_dim))
        base_pow = np.float32(np.float64(1.0) / np.float64(np.float32(rope_theta) ** np.float32(exponent)))
        thetas[i] = base_pow

    # Compute cos/sin tables in float32 matching C++
    cos_table = np.zeros((seq_len, half_dim), dtype=np.float32)
    sin_table = np.zeros((seq_len, half_dim), dtype=np.float32)
    for pos in range(seq_len):
        for j in range(half_dim):
            angle = np.float32(np.float32(pos) * thetas[j])
            cos_table[pos, j] = np.float32(np.cos(np.float64(angle)))
            sin_table[pos, j] = np.float32(np.sin(np.float64(angle)))

    # Use float64 versions for attention computation (higher precision reference)
    cos_table_f64 = cos_table.astype(np.float64)
    sin_table_f64 = sin_table.astype(np.float64)

    def apply_rope(x, cos_t, sin_t):
        """Apply RoPE. Compute in float32 per-element matching C++ scalar loop."""
        x = x.astype(np.float32)
        x1 = x[:half_dim]
        x2 = x[half_dim:]
        out1 = (x1 * cos_t.astype(np.float32) - x2 * sin_t.astype(np.float32)).astype(np.float32)
        out2 = (x1 * sin_t.astype(np.float32) + x2 * cos_t.astype(np.float32)).astype(np.float32)
        return np.concatenate([out1, out2]).astype(np.float32)

    def apply_inverse_rope(x, cos_t, sin_t):
        """Apply inverse RoPE. Compute in float32 matching C++."""
        x = x.astype(np.float32)
        x1 = x[:half_dim]
        x2 = x[half_dim:]
        out1 = (x1 * cos_t.astype(np.float32) + x2 * sin_t.astype(np.float32)).astype(np.float32)
        out2 = (-x1 * sin_t.astype(np.float32) + x2 * cos_t.astype(np.float32)).astype(np.float32)
        return np.concatenate([out1, out2]).astype(np.float32)

    # Inputs
    Q = rand_input((batch, 1, seq_len, q_width))
    K = rand_input((batch, 1, seq_len, kv_width))
    V = rand_input((batch, 1, seq_len, kv_width))

    # Forward pass
    # 1. Apply RoPE to Q and K (element-by-element matching C++)
    Q_rope = Q.copy()
    K_rope = K.copy()
    for b in range(batch):
        for h in range(seq_len):
            for n in range(num_heads_q):
                offset = n * head_dim
                Q_rope[b, 0, h, offset:offset+head_dim] = apply_rope(
                    Q[b, 0, h, offset:offset+head_dim],
                    cos_table[h], sin_table[h])
            for n in range(num_heads_kv):
                offset = n * head_dim
                K_rope[b, 0, h, offset:offset+head_dim] = apply_rope(
                    K[b, 0, h, offset:offset+head_dim],
                    cos_table[h], sin_table[h])

    # 2. Reshape to per-head (float64 for precise attention computation)
    Q_heads = Q_rope.reshape(batch, seq_len, num_heads_q, head_dim).transpose(0, 2, 1, 3).astype(np.float64)
    K_heads = K_rope.reshape(batch, seq_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3).astype(np.float64)
    V_heads = V.reshape(batch, seq_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3).astype(np.float64)

    # 3. Compute attention per Q head with GQA (all in float64)
    scale = 1.0 / np.sqrt(float(head_dim))
    output = np.zeros((batch, num_heads_q, seq_len, head_dim), dtype=np.float64)
    attn_weights = np.zeros((batch, num_heads_q, seq_len, seq_len), dtype=np.float64)

    NEG_FLT_MAX = float(-np.finfo(np.float32).max)

    for b in range(batch):
        for q_head in range(num_heads_q):
            kv_head = q_head // gqa_size
            q_h = Q_heads[b, q_head]
            k_h = K_heads[b, kv_head]
            scores = q_h @ k_h.T * scale

            if is_causal:
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        scores[i, j] = NEG_FLT_MAX

            # Softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            attn_w = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
            attn_weights[b, q_head] = attn_w
            output[b, q_head] = attn_w @ V_heads[b, kv_head]

    # Reshape output and cast to float32
    output_flat = output.transpose(0, 2, 1, 3).reshape(batch, 1, seq_len, q_width).astype(np.float32)

    # Backward pass (incoming derivative = 2.0, in float64)
    d_out_heads = np.full((batch, num_heads_q, seq_len, head_dim), 2.0, dtype=np.float64)

    d_Q_heads = np.zeros_like(Q_heads)  # float64
    d_K_heads = np.zeros((batch, num_heads_kv, seq_len, head_dim), dtype=np.float64)
    d_V_heads = np.zeros((batch, num_heads_kv, seq_len, head_dim), dtype=np.float64)

    for b in range(batch):
        for kv_head in range(num_heads_kv):
            d_k_head = np.zeros((seq_len, head_dim), dtype=np.float64)
            d_v_head = np.zeros((seq_len, head_dim), dtype=np.float64)

            for g in range(gqa_size):
                q_head = kv_head * gqa_size + g
                attn_w = attn_weights[b, q_head]
                d_out = d_out_heads[b, q_head]
                v_h = V_heads[b, kv_head]
                q_h = Q_heads[b, q_head]
                k_h = K_heads[b, kv_head]

                d_attn = d_out @ v_h.T
                d_v_head += attn_w.T @ d_out

                # Softmax backward
                dot_sum = np.sum(d_attn * attn_w, axis=-1, keepdims=True)
                d_scores = attn_w * (d_attn - dot_sum) * scale

                d_Q_heads[b, q_head] = d_scores @ k_h
                d_k_head += d_scores.T @ q_h

            d_K_heads[b, kv_head] = d_k_head
            d_V_heads[b, kv_head] = d_v_head

    # Reshape and cast to float32
    d_Q_flat = d_Q_heads.transpose(0, 2, 1, 3).reshape(batch, 1, seq_len, q_width).astype(np.float32)
    d_K_flat = d_K_heads.transpose(0, 2, 1, 3).reshape(batch, 1, seq_len, kv_width).astype(np.float32)

    # Apply inverse RoPE to d_Q and d_K
    d_Q_out = d_Q_flat.copy()
    d_K_out = d_K_flat.copy()
    for b in range(batch):
        for h in range(seq_len):
            for n in range(num_heads_q):
                offset = n * head_dim
                d_Q_out[b, 0, h, offset:offset+head_dim] = apply_inverse_rope(
                    d_Q_flat[b, 0, h, offset:offset+head_dim],
                    cos_table[h], sin_table[h])
            for n in range(num_heads_kv):
                offset = n * head_dim
                d_K_out[b, 0, h, offset:offset+head_dim] = apply_inverse_rope(
                    d_K_flat[b, 0, h, offset:offset+head_dim],
                    cos_table[h], sin_table[h])

    # d_V doesn't need inverse RoPE (V was not rotated) — cast float64 to float32
    d_V_out = d_V_heads.transpose(0, 2, 1, 3).reshape(batch, 1, seq_len, kv_width).astype(np.float32)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        # 1. Initial weights (none - MHA Core has no trainable weights)
        # 2. Inputs (Q, K, V)
        write_tensor(f, Q)
        write_tensor(f, K)
        write_tensor(f, V)
        # 3. Outputs
        write_tensor(f, output_flat)
        # 4. Gradients (none)
        # 5. Weights (none)
        # 6. Derivatives (dQ, dK, dV)
        write_tensor(f, d_Q_out)
        write_tensor(f, d_K_out)
        write_tensor(f, d_V_out)

    print(f"Generated: {filepath}")
    print(f"  Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"  output shape: {output_flat.shape}")
    print(f"  d_Q sample: {d_Q_out.flat[:5]}")
    print(f"  d_K sample: {d_K_out.flat[:5]}")
    print(f"  d_V sample: {d_V_out.flat[:5]}")

    return filepath


if __name__ == "__main__":
    gen_rms_norm_golden()
    gen_lm_head_golden()
    gen_swiglu_golden()
    gen_tie_word_embedding_golden()
    gen_mha_core_golden()
