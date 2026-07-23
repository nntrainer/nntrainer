from torch import nn


class IFuse(nn.Module):
    @property
    def is_fused(self) -> bool:
        return self._is_fused

    def __init__(self):
        super().__init__()
        self._is_fused = False

    def fuse_operations(self):
        raise NotImplementedError()

    def fuse(self):
        if self.is_fused:
            return self

        self.fuse_operations()
        if hasattr(self, "forward_fuse"):
            self.forward = self.forward_fuse

        self._is_fused = True
        return self
