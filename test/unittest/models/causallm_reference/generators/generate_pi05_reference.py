# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# Tiny pi0.5 reference generator: builds a MINI PaliGemmaWithExpert-style
# model in pure torch (no 14.5GB download, no flax) and reproduces openpi's
# INFERENCE pipeline exactly (PI0Pytorch.sample_actions):
#
#   1. backbone-ALONE prefix pass (bidirectional, plain norms) capturing the
#      post-rope K/V per layer — this is openpi's prefix KV cache;
#   2. per Euler step: expert-ONLY forward (adaRMS cond = time MLP) over the
#      shared cache [backbone K/V | expert suffix K/V], positions
#      Tb + i, action_out_proj -> v_t; x_t += dt * v_t, t: 1 -> 0.
#
# (The joint two-stream concat attention is the TRAINING path only; the
# inference denoise step runs the expert alone against the frozen prefix
# cache — verified against /tmp/openpi pi0_pytorch.py sample_actions.)
#
# The tiny model mirrors openpi's debug_pi05 dummy variant, shrunk:
#   backbone: width 64, 2 layers, mlp 128, 8 heads, 1 kv head, head_dim 16
#   expert:   same geometry, adaRMS norms (Dense(3w)+bias, gate on branch)
#   vision:   SigLIP shape — LN pre-norm blocks with biased FCs, conv patch
#             embed, pos embed, post-LN, projector 32 -> 64
#   vocab 1024, action_dim 8, horizon 4, num_steps 10
#
# Fixture records: all weights (backbone + expert + flow heads + vision),
# the captured prefix K/V per layer (bbN_k / bbN_v, post-rope), the golden
# vision embeddings (vision_out), and per-step v_t + final actions. The C++
# test transplants bbN_k/bbN_v into its shared cache and diffs the Euler
# loop; the 3B e2e covers the real backbone graph separately.

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20261112)
np.random.seed(20261112)

# generators/ sits INSIDE causallm_reference — the fixture dir is the
# sibling "pi05_tiny" next to it.
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   os.pardir, "pi05_tiny"))
os.makedirs(OUT, exist_ok=True)

# ---- tiny hyperparameters -------------------------------------------------
VOCAB = 1024
IMAGE_TOKEN = 1000
D = 64          # backbone + expert width (equal in the tiny fixture)
DEPTH = 2
MLP = 128
HEADS = 8
KV = 1
HD = 16
KVW = KV * HD
VIS_H, VIS_DEPTH, VIS_HEADS = 32, 2, 1
IMG = 8
PATCH = 2
NUM_PATCHES = (IMG // PATCH) ** 2  # 16
PROJ = D
ACTION_DIM = 8
HORIZON = 4
NUM_STEPS = 10
PROMPT_IDS = [2, 7, 9, 5]  # tiny text after the image block
RMS_EPS = 1e-6


def fc(i, o, std=0.02):
    l = nn.Linear(i, o)
    with torch.no_grad():
        l.weight.normal_(0.0, std)
        l.bias.zero_()
    return l


class AdaRMS(nn.Module):
    """openpi's patched GemmaRMSNorm. Plain variant: y = normed*(1+w), no
    gate. adaRMS variant: NO weight param; Dense(3w)+bias on the cond gives
    scale|shift|gate; y = normed*(1+scale)+shift, and the gate multiplies the
    BRANCH output at the residual (x + branch*gate)."""

    def __init__(self, w, cond_dim=None):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.randn(w) * 0.05)
            self.dense = None
        else:
            self.dense = fc(cond_dim, 3 * w, std=0.05)
            self.weight = None

    def forward(self, x, cond=None):
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(var + RMS_EPS)
        if cond is None or self.dense is None:
            return normed * (1.0 + self.weight.float()), None
        sc = self.dense(cond.to(x.dtype))[:, None, :]
        scale, shift, gate = sc.chunk(3, dim=-1)
        return normed * (1.0 + scale) + shift, gate


class LayerNorm(nn.Module):
    """Standard LN with weight+bias (SigLIP), eps 1e-6."""

    def __init__(self, w):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(w) * 0.05)
        self.bias = nn.Parameter(torch.randn(w) * 0.01)

    def forward(self, x):
        m = x.float().mean(-1, keepdim=True)
        v = x.float().var(-1, unbiased=False, keepdim=True)
        y = (x - m) / torch.sqrt(v + 1e-6)
        return y * self.weight + self.bias


def rope_cos_sin(pos, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(pos.float(), inv)
    return torch.cos(f), torch.sin(f)  # [T, hd/2]


def apply_rope(x, cos, sin):
    # x [B, H, T, HD]; non-interleaved (rotate-half) like HF Gemma
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, None]
    s = sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Stream(nn.Module):
    """One Gemma stream (backbone: plain norms; expert: adaRMS)."""

    def __init__(self, w, depth, mlp_dim, heads, kv_heads, hd, adarms=False):
        super().__init__()
        self.w, self.heads, self.kv, self.hd = w, heads, kv_heads, hd
        cd = w if adarms else None
        self.layers = nn.ModuleList()
        for _ in range(depth):
            m = nn.Module()
            m.norm1 = AdaRMS(w, cond_dim=cd)
            m.q = fc(w, heads * hd)
            m.k = fc(w, kv_heads * hd)
            m.v = fc(w, kv_heads * hd)
            m.o = fc(heads * hd, w)
            m.norm2 = AdaRMS(w, cond_dim=cd)
            m.gate = fc(w, mlp_dim)
            m.up = fc(w, mlp_dim)
            m.down = fc(mlp_dim, w)
            self.layers.append(m)
        self.final = AdaRMS(w, cond_dim=cd)

    def mlp(self, x, li):
        l = self.layers[li]
        return l.down(F.gelu(l.gate(x), approximate="tanh") * l.up(x))


class TinyVision(nn.Module):
    """SigLIP-shaped tiny vision: conv patch embed + pos, LN pre-norm blocks
    with biased FCs, post-LN, projector. Single head (C++ mha_core, 1 head,
    head_dim = VIS_H)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, VIS_H, PATCH, PATCH)
        with torch.no_grad():
            self.conv.weight.normal_(0.0, 0.05)
            self.conv.bias.zero_()
        self.pos = nn.Parameter(torch.randn(NUM_PATCHES, VIS_H) * 0.05)
        self.blocks = nn.ModuleList()
        for _ in range(VIS_DEPTH):
            b = nn.Module()
            b.n1 = LayerNorm(VIS_H)
            b.q = fc(VIS_H, VIS_H)
            b.k = fc(VIS_H, VIS_H)
            b.v = fc(VIS_H, VIS_H)
            b.o = fc(VIS_H, VIS_H)
            b.n2 = LayerNorm(VIS_H)
            b.f1 = fc(VIS_H, 2 * VIS_H)
            b.f2 = fc(2 * VIS_H, VIS_H)
            self.blocks.append(b)
        self.post = LayerNorm(VIS_H)
        self.proj = fc(VIS_H, PROJ)

    def forward(self, px):  # px [3, IMG, IMG] in [-1, 1]
        w = self.conv.weight.reshape(VIS_H, 3 * PATCH * PATCH)  # [out, c*k*k]
        p = px.view(3, IMG // PATCH, PATCH, IMG // PATCH, PATCH)
        p = p.permute(1, 3, 2, 4, 0).reshape(NUM_PATCHES, 3 * PATCH * PATCH)
        h = p @ w.T + self.conv.bias + self.pos
        for b in self.blocks:
            n1 = b.n1(h)
            a = (n1 @ b.q(n1).T) / math.sqrt(VIS_H)
            h = h + b.o(a.softmax(-1) @ b.v(n1))
            n2 = b.n2(h)
            h = h + b.f2(F.gelu(b.f1(n2), approximate="tanh"))
        return self.proj(self.post(h))


def make_model():
    m = nn.Module()
    m.vision = TinyVision()
    m.embed = nn.Embedding(VOCAB, D)
    with torch.no_grad():
        m.embed.weight.normal_(0.0, 0.02)
        m.embed.weight[4, 0] = 2.0
    m.back = Stream(D, DEPTH, MLP, HEADS, KV, HD, adarms=False)
    m.expert = Stream(D, DEPTH, MLP, HEADS, KV, HD, adarms=True)
    m.action_in = fc(ACTION_DIM, D)
    m.action_out = fc(D, ACTION_DIM)
    m.time_in = fc(D, D)
    m.time_out = fc(D, D)
    return m


# ---- run ------------------------------------------------------------------
model = make_model()
model.eval()

with torch.no_grad():
    # prefix: bos + 16 image tokens + 4 prompt tokens
    img = torch.tanh(torch.randn(3, IMG, IMG))
    img_embs = model.vision(img)  # [16, D]
    text_embs = model.embed(torch.tensor(PROMPT_IDS)) * math.sqrt(D)
    # openpi order: images first, then language (bos + prompt)
    prefix = torch.cat(
        [img_embs, model.embed(torch.tensor([2])) * math.sqrt(D),
         text_embs], dim=0)  # [Tb, D]
    Tb = prefix.shape[0]
    Te = HORIZON

    noise = torch.randn(Te, ACTION_DIM)

    # ========================================================
    # 1. backbone-ALONE prefix pass: capture post-rope K/V
    #    (openpi sample_actions: inputs_embeds=[prefix, None],
    #    use_cache=True — the expert stream does NOT run here)
    # ========================================================
    pos_b = torch.arange(Tb)
    cos_b, sin_b = rope_cos_sin(pos_b, HD)
    mask_b = torch.zeros(1, 1, Tb, Tb)  # all-allowed (bidirectional prefix)
    x_b = prefix[None]
    bb_k, bb_v = [], []
    for li in range(DEPTH):
        L = model.back.layers[li]
        n1, _ = L.norm1(x_b, None)
        b, t, _ = n1.shape
        q = L.q(n1).view(b, t, HEADS, HD).transpose(1, 2)
        k = L.k(n1).view(b, t, KV, HD).transpose(1, 2)
        v = L.v(n1).view(b, t, KV, HD).transpose(1, 2)
        q = apply_rope(q, cos_b, sin_b)
        k = apply_rope(k, cos_b, sin_b)
        # cache stores POST-rope K, raw V (HF past_key_value convention)
        bb_k.append(k[0].reshape(t, KVW).clone())
        bb_v.append(v[0].reshape(t, KVW).clone())
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(HD) + mask_b).softmax(-1)
        ctx = att @ v  # [B, H, T, HD]
        flat = ctx.transpose(1, 2).reshape(b, t, HEADS * HD)
        x_b = x_b + L.o(flat)
        n2, _ = L.norm2(x_b, None)
        x_b = x_b + model.back.mlp(n2, li)
    xb_final, _ = model.back.final(x_b, None)  # unused at inference

    # ========================================================
    # 2. expert-ONLY Euler loop (openpi denoise_step)
    # ========================================================
    dt = -1.0 / NUM_STEPS
    x_t = noise.clone()
    time = 1.0
    vts = []
    while time >= -dt / 2:
        # sincos time embedding (openpi create_sinusoidal_pos_embedding)
        # C++ side mirrors this exactly: fraction = i/(half-1) (linspace
        # semantics, LAST point exactly 1.0)
        half = D // 2
        frac = torch.arange(half, dtype=torch.float64) / (half - 1)
        period = 4e-3 * (4.0 / 4e-3) ** frac
        ang = (2 * math.pi / period) * time
        t_emb = torch.cat([torch.sin(ang), torch.cos(ang)]).float()
        cond = F.silu(model.time_out(F.silu(model.time_in(t_emb))))

        x_e = model.action_in(x_t)[None]  # [1, Te, D]
        pos_s = torch.arange(Tb, Tb + Te)
        cos_s, sin_s = rope_cos_sin(pos_s, HD)
        for li in range(DEPTH):
            L = model.expert.layers[li]
            n1, g1 = L.norm1(x_e, cond[None])
            b, t, _ = n1.shape
            q = L.q(n1).view(b, t, HEADS, HD).transpose(1, 2)
            k = L.k(n1).view(b, t, KV, HD).transpose(1, 2)
            v = L.v(n1).view(b, t, KV, HD).transpose(1, 2)
            q = apply_rope(q, cos_s, sin_s)
            k = apply_rope(k, cos_s, sin_s)
            # shared cache: frozen backbone K/V + this step's suffix K/V
            # (bb_* stored [Tb, KVW]; expert k/v are [B, KV, Te, HD])
            kbb = bb_k[li].reshape(Tb, KV, HD).transpose(0, 1)[None]
            vbb = bb_v[li].reshape(Tb, KV, HD).transpose(0, 1)[None]
            k_full = torch.cat([kbb, k], dim=2)
            v_full = torch.cat([vbb, v], dim=2)
            # suffix rows attend ALL prefix + ALL suffix (one action block)
            att = ((q @ k_full.transpose(-2, -1)) / math.sqrt(HD)).softmax(-1)
            ctx = att @ v_full
            flat = ctx.transpose(1, 2).reshape(b, t, HEADS * HD)
            out = x_e + L.o(flat) * g1
            n2, g2 = L.norm2(out, cond[None])
            x_e = out + model.expert.mlp(n2, li) * g2
        x_e, _ = model.expert.final(x_e, cond[None])
        v_t = model.action_out(x_e[0])  # [Te, ACTION_DIM]
        vts.append(v_t.clone())
        x_t = x_t + dt * v_t
        time += dt

    final_actions = x_t

# ---- write fixture --------------------------------------------------------
def save(name, t):
    np.asarray(t.detach().float().contiguous().numpy(), dtype="<f4").tofile(
        os.path.join(OUT, name))


names = []


def reg(name, t):
    names.append(name)
    save(name, t)


# vision (C++ SiglipVisionModel load-walk order: conv NHWC, pos, per-layer
# ln1 wb, q/k/v/out wb, ln2 wb, fc1/fc2 wb, post-LN wb, projector wb)
reg("vision_conv_w", model.vision.conv.weight.permute(2, 3, 1, 0))
reg("vision_conv_b", model.vision.conv.bias)
reg("vision_pos", model.vision.pos)
for i, b in enumerate(model.vision.blocks):
    reg(f"vis{i}_ln1_w", b.n1.weight)
    reg(f"vis{i}_ln1_b", b.n1.bias)
    for nm in ("q", "k", "v", "o", "f1", "f2"):
        reg(f"vis{i}_{nm}_w", getattr(b, nm).weight.T)
        reg(f"vis{i}_{nm}_b", getattr(b, nm).bias)
    reg(f"vis{i}_ln2_w", b.n2.weight)
    reg(f"vis{i}_ln2_b", b.n2.bias)
reg("vis_post_w", model.vision.post.weight)
reg("vis_post_b", model.vision.post.bias)
reg("vis_proj_w", model.vision.proj.weight.T)
reg("vis_proj_b", model.vision.proj.bias)
# golden vision output (the C++ test injects this; the 3B e2e covers the
# vision graph itself against HF)
reg("vision_out", img_embs)

reg("embed", model.embed.weight)
for side, stream in (("bb", model.back), ("ex", model.expert)):
    for i, l in enumerate(stream.layers):
        if side == "bb":
            reg(f"{side}{i}_n1_g", l.norm1.weight + 1.0)  # baked
        else:
            reg(f"{side}{i}_n1_dw", l.norm1.dense.weight.T)
            reg(f"{side}{i}_n1_db", l.norm1.dense.bias)
        for nm in ("q", "k", "v", "o", "gate", "up", "down"):
            reg(f"{side}{i}_{nm}_w", getattr(l, nm).weight.T)
        if side == "bb":
            reg(f"{side}{i}_n2_g", l.norm2.weight + 1.0)  # baked
        else:
            reg(f"{side}{i}_n2_dw", l.norm2.dense.weight.T)
            reg(f"{side}{i}_n2_db", l.norm2.dense.bias)
    if side == "bb":
        reg(f"{side}_final_g", stream.final.weight + 1.0)  # baked
    else:
        reg(f"{side}_final_dw", stream.final.dense.weight.T)
        reg(f"{side}_final_db", stream.final.dense.bias)
# captured prefix K/V (post-rope), per layer — the shared-cache head
for i in range(DEPTH):
    reg(f"bb{i}_k", bb_k[i])
    reg(f"bb{i}_v", bb_v[i])
reg("action_in_w", model.action_in.weight.T)
reg("action_in_b", model.action_in.bias)
reg("action_out_w", model.action_out.weight.T)
reg("action_out_b", model.action_out.bias)
reg("time_in_w", model.time_in.weight.T)
reg("time_in_b", model.time_in.bias)
reg("time_out_w", model.time_out.weight.T)
reg("time_out_b", model.time_out.bias)

meta = {
    "model": "pi05_tiny",
    "vocab": VOCAB,
    "image_token_index": IMAGE_TOKEN,
    "hidden_size": D,
    "expert_hidden_size": D,
    "depth": DEPTH,
    "mlp_dim": MLP,
    "heads": HEADS,
    "kv_heads": KV,
    "head_dim": HD,
    "kv_width": KVW,
    "rms_eps": RMS_EPS,
    "rope_theta": 10000.0,
    "time_min_period": 4e-3,
    "time_max_period": 4.0,
    "vision": {"hidden": VIS_H, "layers": VIS_DEPTH, "heads": VIS_HEADS,
               "image_size": IMG, "patch_size": PATCH,
               "num_patches": NUM_PATCHES, "projection_dim": PROJ,
               "ln_eps": 1e-6},
    "action_dim": ACTION_DIM,
    "horizon": HORIZON,
    "num_steps": NUM_STEPS,
    "prompt_ids": PROMPT_IDS,
    "prefix_len": Tb,
    "pixel_values": img.flatten().tolist(),
    "noise": noise.tolist(),
    "per_step_v_t": [v.tolist() for v in vts],
    "final_actions": final_actions.tolist(),
    "weight_order": names,
    "atol_fp32": 2e-3,
    "atol_actions": 1e-3,
}
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", OUT, "with", len(names), "weight records, prefix_len", Tb)
print("final_actions[0][:3]:",
      [round(v, 5) for v in final_actions[0][:3].tolist()])

# ---- assemble the nntrainer bins + config for the C++ tiny test ----
def rd(name):
    # records are written extension-less (save uses os.path.join(OUT, name))
    return np.fromfile(os.path.join(OUT, name), dtype="<f4")

# backbone: bypass layout — layers in graph order, final norm, then the
# tied table TRANSPOSED to [hidden, vocab] as the FC head's own record.
bb = bytearray()
for i in range(DEPTH):
    for nm in (f"bb{i}_n1_g", f"bb{i}_q_w", f"bb{i}_k_w", f"bb{i}_v_w",
               f"bb{i}_o_w", f"bb{i}_n2_g", f"bb{i}_gate_w", f"bb{i}_up_w",
               f"bb{i}_down_w"):
        bb += rd(nm).tobytes()
bb += rd("bb_final_g").tobytes()
emb = rd("embed")
bb += emb.reshape(VOCAB, D).T.copy().tobytes()
with open(os.path.join(OUT, "nntr_pi05_backbone.bin"), "wb") as f:
    f.write(bytes(bb))
with open(os.path.join(OUT, "nntr_pi05_emb.bin"), "wb") as f:
    f.write(emb.astype("<f4").tobytes())

# expert + flow heads in PI05Model::load_weight record order
ex = bytearray()
for i in range(DEPTH):
    for nm in (f"ex{i}_n1_dw", f"ex{i}_n1_db", f"ex{i}_q_w", f"ex{i}_k_w",
               f"ex{i}_v_w", f"ex{i}_o_w", f"ex{i}_n2_dw", f"ex{i}_n2_db",
               f"ex{i}_gate_w", f"ex{i}_up_w", f"ex{i}_down_w"):
        ex += rd(nm).tobytes()
for nm in ("ex_final_dw", "ex_final_db", "action_in_w", "action_in_b",
           "action_out_w", "action_out_b", "time_in_w", "time_in_b",
           "time_out_w", "time_out_b"):
    ex += rd(nm).tobytes()
with open(os.path.join(OUT, "nntr_pi05_expert.bin"), "wb") as f:
    f.write(bytes(ex))

nntr_cfg = {
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_pi05_backbone.bin",
    "model_type": "CausalLM",
    "embedding_dtype": "FP32",
    "fc_layer_dtype": "FP32",
    "lmhead_dtype": "FP32",
    "batch_size": 1,
    "bad_word_ids": [],
    "init_seq_len": int(meta["prefix_len"]) + int(meta["horizon"]) + 4,
    "max_seq_len": int(meta["prefix_len"]) + int(meta["horizon"]) + 12,
    "num_to_generate": 0,
    "use_embedding": True,
    "embedding_bin_path": "nntr_pi05_emb.bin",
    "expert_file_name": "nntr_pi05_expert.bin",
    "expert_hidden_size": D,
    "expert_intermediate_size": MLP,
    "expert_head_dim": HD,
    "expert_num_attention_heads": HEADS,
    "expert_num_key_value_heads": KV,
    "action_dim": ACTION_DIM,
    "action_horizon": HORIZON,
    "num_flow_steps": NUM_STEPS,
    "rms_norm_eps": RMS_EPS,
    "rope_theta": 10000.0,
    "time_min_period": 4e-3,
    "time_max_period": 4.0,
}
with open(os.path.join(OUT, "nntr_config.json"), "w") as fh:
    json.dump(nntr_cfg, fh, indent=2)
tiny_cfg = {
    "architectures": ["PI05Model"],
    "model_type": "paligemma",
    "image_token_index": IMAGE_TOKEN,
    "vocab_size": VOCAB,
    "hidden_size": D,
    "intermediate_size": MLP,
    "num_hidden_layers": DEPTH,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV,
    "head_dim": HD,
    "rms_norm_eps": RMS_EPS,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "expert_hidden_size": D,
    "expert_intermediate_size": MLP,
    "expert_head_dim": HD,
    "expert_num_attention_heads": HEADS,
    "expert_num_key_value_heads": KV,
    "action_dim": ACTION_DIM,
    "action_horizon": HORIZON,
    "num_flow_steps": NUM_STEPS,
    "time_min_period": 4e-3,
    "time_max_period": 4.0,
}
with open(os.path.join(OUT, "config.json"), "w") as fh:
    json.dump(tiny_cfg, fh, indent=2)
with open(os.path.join(OUT, "generation_config.json"), "w") as fh:
    json.dump({"bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
               "do_sample": False}, fh, indent=2)
print("assembled nntr bins + configs in", OUT)
