"""Pattern dataclasses for NNTrainer TorchFX converter.

Defines the data structures used to represent detected model patterns:
  - AttentionPattern: Q/K/V/O projections, optional Q/K norms, SDPA, RoPE
  - FFNPattern: SwiGLU, GELU-FFN, or standard FFN
  - TransformerBlockPattern: Full block structure (norm + op + residual + FFN)
  - ModelStructure: Complete model overview
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttentionPattern:
    """Detected multi-head attention pattern."""
    block_idx: int
    q_proj: str = ""
    k_proj: str = ""
    v_proj: str = ""
    o_proj: str = ""
    q_norm: str = ""
    k_norm: str = ""
    sdpa: str = ""
    attention_type: str = "mha"             # "mha", "gqa", "mqa"
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    has_rope: bool = False
    has_qk_norm: bool = False
    layer_names: list = field(default_factory=list)

    @property
    def scope(self):
        """Common HF module scope (e.g. 'model.layers.0.self_attn')."""
        if self.q_proj:
            return ""
        return ""


@dataclass
class FFNPattern:
    """Detected feed-forward network pattern."""
    block_idx: int
    ffn_type: str = "standard"              # "swiglu", "gelu_ffn", "standard"
    gate_proj: str = ""
    up_proj: str = ""
    down_proj: str = ""
    activation: str = ""
    gate_multiply: str = ""
    intermediate_size: int = 0
    layer_names: list = field(default_factory=list)


@dataclass
class SSMPattern:
    """Detected State Space Model (Mamba) pattern."""
    block_idx: int
    in_proj: str = ""          # Linear: input projection (expand dim)
    conv1d: str = ""           # Conv1d: causal temporal convolution
    x_proj: str = ""           # Linear: compute selection params (B, C)
    dt_proj: str = ""          # Linear: compute time step delta
    out_proj: str = ""         # Linear: output projection (contract dim)
    norm: str = ""             # Optional inner normalization
    ssm_type: str = "mamba"    # "mamba", "mamba2", etc.
    state_size: int = 0        # N: SSM state dimension
    conv_kernel: int = 0       # d_conv: causal conv kernel size
    expand: int = 0            # expansion factor (inner_dim / hidden_dim)
    dt_rank: int = 0           # rank of dt projection
    layer_names: list = field(default_factory=list)


@dataclass
class TransformerBlockPattern:
    """Detected transformer block."""
    block_idx: int
    block_role: str = ""                    # "encoder", "decoder", or ""
    pre_attn_norm: str = ""
    attention: Optional[AttentionPattern] = None
    post_attn_norm: str = ""
    attn_residual: str = ""
    pre_ffn_norm: str = ""
    ffn: Optional[FFNPattern] = None
    post_ffn_norm: str = ""
    ffn_residual: str = ""
    norm_type: str = "pre_norm"             # "pre_norm" or "post_norm"
    ssm: Optional[SSMPattern] = None
    cross_attention: Optional[AttentionPattern] = None
    cross_attn_norm: str = ""
    cross_attn_residual: str = ""
    operator_type: str = ""                 # e.g. "attention", "conv", "mixer"
    operator_scope: str = ""
    operator_layers: list = field(default_factory=list)


@dataclass
class ModelStructure:
    """Full model structure overview."""
    arch_type: str = ""
    model_type: str = ""
    embedding: str = ""
    blocks: list = field(default_factory=list)
    lm_head: str = ""
    final_norm: str = ""
    tie_word_embeddings: bool = False
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    rope_theta: float = 0.0
    norm_eps: float = 0.0
    max_position_embeddings: int = 0
    num_encoder_layers: int = 0
    num_decoder_layers: int = 0
    conv_l_cache: int = 0
    external_kv_cache: bool = False
    # SSM / Mamba config
    ssm_state_size: int = 0       # N: SSM state dimension
    ssm_conv_kernel: int = 0      # d_conv: causal conv kernel size
    ssm_expand: int = 0           # expansion factor
    ssm_dt_rank: int = 0          # rank of dt projection

    @property
    def encoder_blocks(self):
        return [b for b in self.blocks if b.block_role == "encoder"]

    @property
    def decoder_blocks(self):
        return [b for b in self.blocks if b.block_role == "decoder"]

    def summary(self):
        print(f"\n{'='*70}")
        print(f"DETECTED MODEL STRUCTURE")
        print(f"{'='*70}")
        print(f"Architecture: {self.arch_type} ({self.model_type})")
        print(f"Config: hidden={self.hidden_size}, layers={self.num_layers}, "
              f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
              f"head_dim={self.head_dim}")
        print(f"Intermediate: {self.intermediate_size}, "
              f"vocab: {self.vocab_size}")
        if self.rope_theta:
            print(f"RoPE theta: {self.rope_theta}")
        if self.ssm_state_size:
            print(f"SSM: state_size={self.ssm_state_size}, "
                  f"conv_kernel={self.ssm_conv_kernel}, "
                  f"expand={self.ssm_expand}, dt_rank={self.ssm_dt_rank}")
        print(f"Embedding: {self.embedding}")
        if self.final_norm:
            print(f"Final norm: {self.final_norm}")
        if self.lm_head:
            print(f"LM head: {self.lm_head} "
                  f"(tied={self.tie_word_embeddings})")

        for block in self.blocks:
            print_block(block)
        print(f"{'='*70}")


def print_block(block):
    """Print a single transformer block."""
    role = f" [{block.block_role}]" if block.block_role else ""
    print(f"\n  Block {block.block_idx}{role}:")
    print(f"    Norm type: {block.norm_type}")
    if block.pre_attn_norm:
        print(f"    Pre-attn norm: {block.pre_attn_norm}")
    if block.post_attn_norm:
        print(f"    Post-attn norm: {block.post_attn_norm}")
    if block.attention:
        attn = block.attention
        print(f"    Attention ({attn.attention_type}):")
        print(f"      Q: {attn.q_proj}, K: {attn.k_proj}, V: {attn.v_proj}")
        if attn.has_qk_norm:
            print(f"      Q-norm: {attn.q_norm}, K-norm: {attn.k_norm}")
        print(f"      O: {attn.o_proj}")
        print(f"      heads={attn.num_heads}, kv_heads={attn.num_kv_heads}, "
              f"head_dim={attn.head_dim}")
        if attn.has_rope:
            print(f"      RoPE: yes")
    if block.attn_residual:
        print(f"    Attn residual: {block.attn_residual}")

    if block.cross_attention:
        xattn = block.cross_attention
        print(f"    Cross-attention ({xattn.attention_type}):")
        print(f"      Q: {xattn.q_proj}, K: {xattn.k_proj}, V: {xattn.v_proj}")
        print(f"      O: {xattn.o_proj}")

    if block.ssm:
        ssm = block.ssm
        print(f"    SSM ({ssm.ssm_type}):")
        print(f"      in_proj: {ssm.in_proj}, conv1d: {ssm.conv1d}")
        print(f"      x_proj: {ssm.x_proj}, dt_proj: {ssm.dt_proj}")
        print(f"      out_proj: {ssm.out_proj}")
        print(f"      state_size={ssm.state_size}, conv_kernel={ssm.conv_kernel}, "
              f"expand={ssm.expand}, dt_rank={ssm.dt_rank}")

    if block.pre_ffn_norm:
        print(f"    Pre-FFN norm: {block.pre_ffn_norm}")
    if block.post_ffn_norm:
        print(f"    Post-FFN norm: {block.post_ffn_norm}")
    if block.ffn:
        ffn = block.ffn
        print(f"    FFN ({ffn.ffn_type}):")
        if ffn.ffn_type in ("swiglu", "geglu") or ffn.ffn_type.startswith("gated_"):
            print(f"      gate: {ffn.gate_proj}, up: {ffn.up_proj}, "
                  f"down: {ffn.down_proj}")
        else:
            print(f"      fc1: {ffn.up_proj}, fc2: {ffn.down_proj}")
        print(f"      intermediate_size={ffn.intermediate_size}")
    if block.ffn_residual:
        print(f"    FFN residual: {block.ffn_residual}")
