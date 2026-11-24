# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file model.py
# @brief Semantic model IR.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Semantic model IR.

This is the architecture-level source of truth introduced between the
raw module-tree walk (GenericFxParser) and the nntrainer target graph.
An ArchitectureAdapter (api/adapters) builds one of these directly from
the HF config + AutoModel.from_config(config) module tree; the
nntrainer lowering stage (api/lowering/nntrainer) consumes it to build
the target graph and, from there, the generated C++.

Every dataclass here is plain data (no torch tensors, no framework
objects) specifically so it round-trips through JSON and can live in
PipelineState / state.json without special-casing the checkpoint code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


def _dc_to_dict(obj):
    """asdict() that also tolerates None / already-plain values."""
    if obj is None:
        return None
    return asdict(obj)


@dataclass
class ProjectionIR:
    """A linear projection: source name, output size and bias."""
    source_name: str
    output_size: int
    bias: bool = False
    weight_name: str = ""
    input_size: Optional[int] = None

    def to_dict(self) -> dict:
        """The projection as JSON-ready data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ProjectionIR"]:
        """A projection from to_dict() data, or None."""
        return cls(**data) if data else None


@dataclass
class NormIR:
    """A normalisation layer: its type and feature size."""
    source_name: str
    norm_type: str
    feature_size: int
    epsilon: float
    reshaped: bool = False

    def to_dict(self) -> dict:
        """The norm as JSON-ready data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["NormIR"]:
        """A norm from to_dict() data, or None."""
        return cls(**data) if data else None


@dataclass
# pylint: disable-next=too-many-instance-attributes
class AttentionIR:
    """An attention block: its projections, heads and optional Q/K norms."""
    source_name: str

    q_proj: ProjectionIR
    k_proj: ProjectionIR
    v_proj: ProjectionIR
    o_proj: ProjectionIR

    num_heads: int
    num_kv_heads: int
    head_dim: int

    q_norm: Optional[NormIR] = None
    k_norm: Optional[NormIR] = None

    rope_theta: Optional[float] = None
    max_position_embeddings: Optional[int] = None
    sliding_window: Optional[int] = None

    causal: bool = True
    use_kv_cache: bool = True

    def to_dict(self) -> dict:
        """The attention block as JSON-ready data."""
        return {
            "source_name": self.source_name,
            "q_proj": self.q_proj.to_dict(),
            "k_proj": self.k_proj.to_dict(),
            "v_proj": self.v_proj.to_dict(),
            "o_proj": self.o_proj.to_dict(),
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "q_norm": _dc_to_dict(self.q_norm),
            "k_norm": _dc_to_dict(self.k_norm),
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
            "causal": self.causal,
            "use_kv_cache": self.use_kv_cache,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionIR":
        """An attention block from to_dict() data."""
        return cls(
            source_name=data["source_name"],
            q_proj=ProjectionIR.from_dict(data["q_proj"]),
            k_proj=ProjectionIR.from_dict(data["k_proj"]),
            v_proj=ProjectionIR.from_dict(data["v_proj"]),
            o_proj=ProjectionIR.from_dict(data["o_proj"]),
            num_heads=data["num_heads"],
            num_kv_heads=data["num_kv_heads"],
            head_dim=data["head_dim"],
            q_norm=NormIR.from_dict(data.get("q_norm")),
            k_norm=NormIR.from_dict(data.get("k_norm")),
            rope_theta=data.get("rope_theta"),
            max_position_embeddings=data.get("max_position_embeddings"),
            sliding_window=data.get("sliding_window"),
            causal=data.get("causal", True),
            use_kv_cache=data.get("use_kv_cache", True),
        )


@dataclass
class MLPIR:
    """A feed-forward block: up, down and optional gate projections."""
    source_name: str
    up_proj: ProjectionIR
    down_proj: ProjectionIR
    activation: str
    gated: bool
    gate_proj: Optional[ProjectionIR] = None

    def to_dict(self) -> dict:
        """The feed-forward block as JSON-ready data."""
        return {
            "source_name": self.source_name,
            "up_proj": self.up_proj.to_dict(),
            "down_proj": self.down_proj.to_dict(),
            "activation": self.activation,
            "gated": self.gated,
            "gate_proj": _dc_to_dict(self.gate_proj),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MLPIR":
        """A feed-forward block from to_dict() data."""
        return cls(
            source_name=data["source_name"],
            up_proj=ProjectionIR.from_dict(data["up_proj"]),
            down_proj=ProjectionIR.from_dict(data["down_proj"]),
            activation=data["activation"],
            gated=data["gated"],
            gate_proj=ProjectionIR.from_dict(data.get("gate_proj")),
        )


@dataclass
class DecoderLayerIR:
    """One decoder layer: its norms, attention and feed-forward block."""
    index: int
    input_norm: NormIR
    attention: AttentionIR
    post_attention_norm: NormIR
    mlp: MLPIR

    def to_dict(self) -> dict:
        """The decoder layer as JSON-ready data."""
        return {
            "index": self.index,
            "input_norm": self.input_norm.to_dict(),
            "attention": self.attention.to_dict(),
            "post_attention_norm": self.post_attention_norm.to_dict(),
            "mlp": self.mlp.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecoderLayerIR":
        """A decoder layer from to_dict() data."""
        return cls(
            index=data["index"],
            input_norm=NormIR.from_dict(data["input_norm"]),
            attention=AttentionIR.from_dict(data["attention"]),
            post_attention_norm=NormIR.from_dict(data["post_attention_norm"]),
            mlp=MLPIR.from_dict(data["mlp"]),
        )

    def structural_signature(self) -> tuple:
        """A hashable summary of everything that affects codegen shape,
        deliberately excluding `index` and per-layer weight *names* --
        two layers with the same signature can share one generated C++
        function (see converters/cpp_generator.py CAUSALLM_COMPONENT emitter).
        Errs on the side of including a field rather than omitting it:
        folding two structurally-different layers into one reusable
        function would silently generate wrong C++, whereas failing to
        fold two genuinely-identical layers only costs a few extra
        lines of unrolled code."""
        attn = self.attention
        mlp_ir = self.mlp
        return (
            attn.q_proj.output_size, attn.k_proj.output_size,
            attn.v_proj.output_size, attn.o_proj.output_size,
            attn.q_proj.input_size, attn.k_proj.input_size,
            attn.v_proj.input_size, attn.o_proj.input_size,
            attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias, attn.o_proj.bias,
            attn.num_heads, attn.num_kv_heads, attn.head_dim,
            attn.q_norm is not None, attn.k_norm is not None,
            attn.rope_theta, attn.max_position_embeddings, attn.sliding_window,
            attn.causal, attn.use_kv_cache,
            self.input_norm.norm_type, self.input_norm.epsilon, self.input_norm.reshaped,
            self.post_attention_norm.norm_type, self.post_attention_norm.epsilon,
            self.post_attention_norm.reshaped,
            mlp_ir.gated, mlp_ir.activation,
            mlp_ir.gate_proj.output_size if mlp_ir.gate_proj else None,
            mlp_ir.gate_proj.input_size if mlp_ir.gate_proj else None,
            mlp_ir.gate_proj.bias if mlp_ir.gate_proj else None,
            mlp_ir.up_proj.output_size, mlp_ir.up_proj.input_size, mlp_ir.up_proj.bias,
            mlp_ir.down_proj.output_size, mlp_ir.down_proj.input_size, mlp_ir.down_proj.bias,
        )


@dataclass
# pylint: disable-next=too-many-instance-attributes
class CausalLMIR:
    """A whole causal language model: embedding, decoder layers, final norm and LM head."""
    architecture: str
    hidden_size: int
    vocab_size: int
    num_layers: int

    embedding_name: str
    decoder_layers: list[DecoderLayerIR]
    final_norm: NormIR
    lm_head_name: str

    tied_embeddings: bool = False

    def to_dict(self) -> dict:
        """The model as JSON-ready data."""
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "embedding_name": self.embedding_name,
            "decoder_layers": [l.to_dict() for l in self.decoder_layers],
            "final_norm": self.final_norm.to_dict(),
            "lm_head_name": self.lm_head_name,
            "tied_embeddings": self.tied_embeddings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CausalLMIR":
        """A model from to_dict() data."""
        return cls(
            architecture=data["architecture"],
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            num_layers=data["num_layers"],
            embedding_name=data["embedding_name"],
            decoder_layers=[DecoderLayerIR.from_dict(l) for l in data["decoder_layers"]],
            final_norm=NormIR.from_dict(data["final_norm"]),
            lm_head_name=data["lm_head_name"],
            tied_embeddings=data.get("tied_embeddings", False),
        )

    def uniform_layer_signature(self) -> Optional[tuple]:
        """Returns the shared structural signature if every decoder layer
        is structurally identical, else None. Drives whether the C++
        generator can emit one reusable createDecoderLayer() + a loop
        instead of unrolling every layer (see cpp_generator.py)."""
        if not self.decoder_layers:
            return None
        sigs = {l.structural_signature() for l in self.decoder_layers}
        return next(iter(sigs)) if len(sigs) == 1 else None
