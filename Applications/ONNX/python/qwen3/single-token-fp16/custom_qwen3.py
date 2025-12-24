"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>

@file custom_qwen3.py
@date 25 December 2025
@brief This is a Qwen3 1.7B modeling file that is compatible with Hugging Face and supports conversion
       from PyTorch to ONNX to NNTrainer. It does not include masking and caching for token generation.
@note This script has been tested with transformers version 4.55.0 and PyTorch version 2.8.0

@author Sumon Nath <sumon.nath@samsung.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        # with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states, eps):
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(3,keepdim=True) 
        # variance = hidden_states.pow(2).mean(3,keepdim=True) 
        std_dev = torch.sqrt(variance + eps)
        hidden_states = hidden_states * torch.pow(std_dev, -1)
        return hidden_states.to(input_dtype) * self.weight


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : x.shape[-1]]
    return torch.cat((-x2, x1), dim=3)  # -1 changed to 3


def apply_rotary_pos_emb(q, k, cos, sin):

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = torch.cat([hidden_states] * n_rep, dim=2)
    hidden_states = hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )
    return hidden_states


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim
        )  # thus post q_norm does not need reshape

    def forward(self, hidden_states, cos, sin, eps):

        q_hidden_shape = (1, 1, self.config.num_attention_heads, self.head_dim)
        kv_hidden_shape = (1, 1, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(q_hidden_shape), eps
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(kv_hidden_shape), eps
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        # The above line is replaced by mul operation and reducesum.
        # This was done to overcome an issue related with the above matmul op while inferencing in nntrainer
        # This only working in the case single token inference.

        attn_weights = query_states.transpose(1, 2) * key_states.transpose(1, 2)
        attn_weights = torch.sum(attn_weights, dim=3, keepdim=True)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)

        attn_output = value_states.transpose(1, 2) * attn_weights

        # Line above replaces the below two.
        # attn_output = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(1, 1, 1, self.config.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size)

    def forward(self, hidden_states, cos, sin, eps):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, eps)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, cos, sin, eps)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states, eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.register_buffer("vocab_weights", self.embed_tokens.weight)
        self.register_buffer(
            "broadcast", torch.ones(1, config.hidden_size, dtype=torch.int64)
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(config.hidden_size)

    def forward(self, input_ids, cos, sin, eps):

        # hidden_states_new = self.embed_tokens(input_ids)

        # nn.embedding is replaced by gather op for easier tracing. Hence input is broadcasted to (1,1,1,self.config,hidden_size).
        input_ids = self.broadcast * input_ids
        hidden_states = torch.gather(self.vocab_weights, 0, input_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states.reshape(1, 1, 1, self.config.hidden_size), cos, sin, eps
            )

        hidden_states = self.norm(hidden_states, eps)
        return hidden_states


class NNTrainerQwen3ForCausalLM(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, cos, sin, eps):
        hidden_states = self.model(input_ids, cos, sin, eps)

        logits = self.lm_head(hidden_states)

        return logits
