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

from dataclasses import dataclass, field, asdict
from typing import Optional


def _dc_to_dict(obj):
    """asdict() that also tolerates None / already-plain values."""
    if obj is None:
        return None
    return asdict(obj)


@dataclass
class ProjectionIR:
    source_name: str
    output_size: int
    bias: bool = False
    weight_name: str = ""
    input_size: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["ProjectionIR"]:
        return cls(**d) if d else None


@dataclass
class NormIR:
    source_name: str
    norm_type: str
    feature_size: int
    epsilon: float
    reshaped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["NormIR"]:
        return cls(**d) if d else None


@dataclass
class AttentionIR:
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
    def from_dict(cls, d: dict) -> "AttentionIR":
        return cls(
            source_name=d["source_name"],
            q_proj=ProjectionIR.from_dict(d["q_proj"]),
            k_proj=ProjectionIR.from_dict(d["k_proj"]),
            v_proj=ProjectionIR.from_dict(d["v_proj"]),
            o_proj=ProjectionIR.from_dict(d["o_proj"]),
            num_heads=d["num_heads"],
            num_kv_heads=d["num_kv_heads"],
            head_dim=d["head_dim"],
            q_norm=NormIR.from_dict(d.get("q_norm")),
            k_norm=NormIR.from_dict(d.get("k_norm")),
            rope_theta=d.get("rope_theta"),
            max_position_embeddings=d.get("max_position_embeddings"),
            sliding_window=d.get("sliding_window"),
            causal=d.get("causal", True),
            use_kv_cache=d.get("use_kv_cache", True),
        )


@dataclass
class MLPIR:
    source_name: str
    up_proj: ProjectionIR
    down_proj: ProjectionIR
    activation: str
    gated: bool
    gate_proj: Optional[ProjectionIR] = None

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "up_proj": self.up_proj.to_dict(),
            "down_proj": self.down_proj.to_dict(),
            "activation": self.activation,
            "gated": self.gated,
            "gate_proj": _dc_to_dict(self.gate_proj),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MLPIR":
        return cls(
            source_name=d["source_name"],
            up_proj=ProjectionIR.from_dict(d["up_proj"]),
            down_proj=ProjectionIR.from_dict(d["down_proj"]),
            activation=d["activation"],
            gated=d["gated"],
            gate_proj=ProjectionIR.from_dict(d.get("gate_proj")),
        )


@dataclass
class DecoderLayerIR:
    index: int
    input_norm: NormIR
    attention: AttentionIR
    post_attention_norm: NormIR
    mlp: MLPIR

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "input_norm": self.input_norm.to_dict(),
            "attention": self.attention.to_dict(),
            "post_attention_norm": self.post_attention_norm.to_dict(),
            "mlp": self.mlp.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderLayerIR":
        return cls(
            index=d["index"],
            input_norm=NormIR.from_dict(d["input_norm"]),
            attention=AttentionIR.from_dict(d["attention"]),
            post_attention_norm=NormIR.from_dict(d["post_attention_norm"]),
            mlp=MLPIR.from_dict(d["mlp"]),
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
        a = self.attention
        m = self.mlp
        return (
            a.q_proj.output_size, a.k_proj.output_size,
            a.v_proj.output_size, a.o_proj.output_size,
            a.q_proj.input_size, a.k_proj.input_size,
            a.v_proj.input_size, a.o_proj.input_size,
            a.q_proj.bias, a.k_proj.bias, a.v_proj.bias, a.o_proj.bias,
            a.num_heads, a.num_kv_heads, a.head_dim,
            a.q_norm is not None, a.k_norm is not None,
            a.rope_theta, a.max_position_embeddings, a.sliding_window,
            a.causal, a.use_kv_cache,
            self.input_norm.norm_type, self.input_norm.epsilon, self.input_norm.reshaped,
            self.post_attention_norm.norm_type, self.post_attention_norm.epsilon,
            self.post_attention_norm.reshaped,
            m.gated, m.activation,
            m.gate_proj.output_size if m.gate_proj else None,
            m.gate_proj.input_size if m.gate_proj else None,
            m.gate_proj.bias if m.gate_proj else None,
            m.up_proj.output_size, m.up_proj.input_size, m.up_proj.bias,
            m.down_proj.output_size, m.down_proj.input_size, m.down_proj.bias,
        )


@dataclass
class CausalLMIR:
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
    def from_dict(cls, d: dict) -> "CausalLMIR":
        return cls(
            architecture=d["architecture"],
            hidden_size=d["hidden_size"],
            vocab_size=d["vocab_size"],
            num_layers=d["num_layers"],
            embedding_name=d["embedding_name"],
            decoder_layers=[DecoderLayerIR.from_dict(l) for l in d["decoder_layers"]],
            final_norm=NormIR.from_dict(d["final_norm"]),
            lm_head_name=d["lm_head_name"],
            tied_embeddings=d.get("tied_embeddings", False),
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
