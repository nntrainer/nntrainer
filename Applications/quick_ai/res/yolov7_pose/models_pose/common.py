import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReOrg(nn.Module):
    def forward(self, x):
        return torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
            1,
        )


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x[..., None] ** 2, dim=2, keepdim=True)) * self.scale
        norm = torch.squeeze(norm, 3)
        return x / norm.clamp(min=self.eps) * self.g


class RTMCCBlock(nn.Module):
    def __init__(
        self,
        num_token,
        in_token_dims,
        out_token_dims,
        expansion_factor=2,
        s=128,
        eps=1e-5,
        dropout_rate=0.0,
        drop_path=0.0,
        attn_type="self-attn",
        act_fn="SiLU",
        bias=False,
        use_rel_bias=True,
        pos_enc=False,
    ):
        super().__init__()
        self.s = s
        self.num_token = num_token
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = nn.SiLU(True)
        self.shortcut = True
        self.res_scale = Scale(in_token_dims)
        self.sqrt_s = math.sqrt(s)

    def _forward(self, inputs):
        x = self.ln(inputs)
        uv = self.act_fn(self.uv(x))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = torch.unbind(base, dim=2)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        return self.o(u * torch.bmm(kernel, v))

    def forward(self, x):
        return self.res_scale(x) + self.drop_path(self._forward(x))
