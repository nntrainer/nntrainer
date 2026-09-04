from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEdge:
    """
    Represents a directed edge in the Compiler IR graph.
    """

    source: str
    target: str

    metadata: dict[str, Any] = field(default_factory=dict)

    # Dataflow information
    tensor_name: str = ""
    tensor_shape: list[int] = field(default_factory=list)
    dtype: str = ""

    # Memory planning (future)
    buffer_id: str = ""
    memory_size: int = 0

    # Dependency analysis
    control_dependency: bool = False
    data_dependency: bool = True

    # Optimization flags
    fused: bool = False
    removed: bool = False

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "metadata": self.metadata,
            "tensor_name": self.tensor_name,
            "tensor_shape": self.tensor_shape,
            "dtype": self.dtype,
            "buffer_id": self.buffer_id,
            "memory_size": self.memory_size,
            "control_dependency": self.control_dependency,
            "data_dependency": self.data_dependency,
            "fused": self.fused,
            "removed": self.removed,
        }

    def __repr__(self):
        return (
            f"GraphEdge("
            f"{self.source} -> {self.target}, "
            f"dtype={self.dtype})"
        )