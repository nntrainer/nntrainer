# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>

## @file weight_converter.py
## @brief Weight conversion for the ScreenAI caption v4.0.0 checkpoint bundle.
##
## v4.0.0 ships the model as THREE separate artifacts instead of the single
## combined VisionEncoderDecoder checkpoint used by caption-s02:
##
##   siglip2-base-patch16-384/model.safetensors  keys "vision_model.*"
##   best/decoder/model.safetensors              keys "bert.*" / "cls.*"
##   best/encoder_to_decoder.pt                  {"weight": [512,768], "bias": [512]}
##
## The nntrainer-side tensor order/naming is identical to caption-s02, so the
## emitted files are drop-in for the existing SigLIP2VisionEncoder / BertDecoder
## runtime; only the source key prefixes and the dimensions differ.
##
## Architecture (from best/model_config.json + best/decoder/config.json):
##   encoder  SigLIP2 ViT-B/16-384, 12 layers, hidden 768, 576 patches, no CLS
##   connect  Linear(768 -> 512)
##   decoder  BertLMHeadModel, 4 layers, hidden 512, 8 heads, ffn 2048,
##            vocab 30522, cross-attends over 576 encoder tokens
## @author Seungbaek Hong <sb92.hong@samsung.com>

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SAFETENSORS_DTYPE_MAP = {"float32": "F32"}

ENC_LAYERS = 12  # SigLIP2 ViT-B/16
DEC_LAYERS = 4   # BERT small decoder

EXPECTED_ENC_TENSORS = 199
EXPECTED_DEC_TENSORS = 114


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert torch tensor to contiguous numpy array."""
    if transpose:
        tensor = tensor.permute(1, 0)
    return np.ascontiguousarray(tensor.detach().cpu().float().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_safetensors(weights, output_path, dtype):
    """Write weights to a safetensors file (format written directly)."""
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        raw_buffers.append(data)
        offset += len(data)

    tensor_meta["__metadata__"] = metadata
    header = json.dumps(tensor_meta, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for data in raw_buffers:
            f.write(data)

    total = sum(len(b) for b in raw_buffers)
    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {total / 1e9:.4f} GB")


def save_bin(weights, output_path):
    """Write weights to nntrainer raw binary file (order-defined)."""
    with open(output_path, "wb") as f:
        for _name, arr in weights:
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            arr.tofile(f)
    total_bytes = sum(arr.nbytes for _, arr in weights)
    print(f"Saved binary: {output_path}")
    print(f"Tensor data size: {total_bytes / 1e9:.4f} GB")


# ---------------------------------------------------------------------------
# Encoder: siglip2-base-patch16-384/model.safetensors + encoder_to_decoder.pt
#
# Order (must match SigLIP2VisionEncoder weight creation order):
#  1-2. patch_embedding weight(OIHW, no transpose), bias
#  3.   position_embedding [576,768] -> [1,1,576,768]
#  4.   12 x { ln1 g/b, wq(T)/b, wk(T)/b, wv(T)/b, out(T)/b,
#              ln2 g/b, fc1(T)/b, fc2(T)/b }
#  5.   post_layernorm weight, bias
#  6.   enc_to_dec_proj weight(T), bias      <- from the separate .pt
# ---------------------------------------------------------------------------

def collect_encoder(sd, proj, dtype):
    """Return ordered list of (nntr_name, ndarray) for the encoder file."""
    weights = []

    def add(name, tensor, transpose=False):
        weights.append((name, tensor_to_numpy(tensor, dtype, transpose=transpose)))

    vp = "vision_model."

    # 1-2. Patch embedding (Conv2D - keep OIHW, no transpose). conv2d names its
    # kernel "filter" (not "weight"); the safetensors loader matches by name and
    # silently skips misses, so these strings must stay exact.
    add("patch_embed_conv:filter",
        sd[f"{vp}embeddings.patch_embedding.weight"], transpose=False)
    add("patch_embed_conv:bias",
        sd[f"{vp}embeddings.patch_embedding.bias"])

    # 3. Position embedding [576,768] -> [1,1,576,768]
    pos = sd[f"{vp}embeddings.position_embedding.weight"]
    add("pos_embedding:pos_embedding",
        pos.unsqueeze(0).unsqueeze(0), transpose=False)

    # 4. Transformer layers
    for i in range(ENC_LAYERS):
        lp = f"{vp}encoder.layers.{i}."
        pfx = f"enc_layer{i}"

        add(f"{pfx}_ln1:gamma", sd[f"{lp}layer_norm1.weight"])
        add(f"{pfx}_ln1:beta", sd[f"{lp}layer_norm1.bias"])

        add(f"{pfx}_wq:weight", sd[f"{lp}self_attn.q_proj.weight"], transpose=True)
        add(f"{pfx}_wq:bias", sd[f"{lp}self_attn.q_proj.bias"])
        add(f"{pfx}_wk:weight", sd[f"{lp}self_attn.k_proj.weight"], transpose=True)
        add(f"{pfx}_wk:bias", sd[f"{lp}self_attn.k_proj.bias"])
        add(f"{pfx}_wv:weight", sd[f"{lp}self_attn.v_proj.weight"], transpose=True)
        add(f"{pfx}_wv:bias", sd[f"{lp}self_attn.v_proj.bias"])
        add(f"{pfx}_out:weight", sd[f"{lp}self_attn.out_proj.weight"], transpose=True)
        add(f"{pfx}_out:bias", sd[f"{lp}self_attn.out_proj.bias"])

        add(f"{pfx}_ln2:gamma", sd[f"{lp}layer_norm2.weight"])
        add(f"{pfx}_ln2:beta", sd[f"{lp}layer_norm2.bias"])

        add(f"{pfx}_fc1:weight", sd[f"{lp}mlp.fc1.weight"], transpose=True)
        add(f"{pfx}_fc1:bias", sd[f"{lp}mlp.fc1.bias"])
        add(f"{pfx}_fc2:weight", sd[f"{lp}mlp.fc2.weight"], transpose=True)
        add(f"{pfx}_fc2:bias", sd[f"{lp}mlp.fc2.bias"])

    # 5. Post-LayerNorm
    add("post_ln:gamma", sd[f"{vp}post_layernorm.weight"])
    add("post_ln:beta", sd[f"{vp}post_layernorm.bias"])

    # 6. Encoder-to-decoder projection (stored with the encoder file).
    # v4.0.0 keeps this in a standalone encoder_to_decoder.pt.
    add("enc_to_dec_proj:weight", proj["weight"], transpose=True)
    add("enc_to_dec_proj:bias", proj["bias"])

    return weights


# ---------------------------------------------------------------------------
# Decoder: best/decoder/model.safetensors
#
# Order (must match BertDecoder weight creation order):
#  1-5. word/pos/type embeddings, emb LayerNorm g/b
#  6.   4 x { self q/k/v (T)+b, self out(T)+b, self ln g/b,
#             cross q/k/v (T)+b, cross out(T)+b, cross ln g/b,
#             ffn inter(T)+b, ffn out(T)+b, ffn ln g/b }
#  7.   lmhead dense(T)+b, lmhead ln g/b, lmhead bias [30522]
#       (the vocab projection is TIED to word_embeddings - not re-saved)
# ---------------------------------------------------------------------------

def collect_decoder(sd, dtype):
    """Return ordered list of (nntr_name, ndarray) for the decoder file."""
    weights = []

    def add(name, tensor, transpose=False):
        weights.append((name, tensor_to_numpy(tensor, dtype, transpose=transpose)))

    bp = "bert."

    # 1-5. Embeddings. embedding_layer names its table "Embedding".
    add("word_emb:Embedding", sd[f"{bp}embeddings.word_embeddings.weight"])
    add("pos_emb:Embedding", sd[f"{bp}embeddings.position_embeddings.weight"])
    add("type_emb:Embedding", sd[f"{bp}embeddings.token_type_embeddings.weight"])
    add("emb_ln:gamma", sd[f"{bp}embeddings.LayerNorm.weight"])
    add("emb_ln:beta", sd[f"{bp}embeddings.LayerNorm.bias"])

    # 6. Decoder layers
    for i in range(DEC_LAYERS):
        lp = f"{bp}encoder.layer.{i}."
        pfx = f"dec_layer{i}"

        add(f"{pfx}_self_q:weight", sd[f"{lp}attention.self.query.weight"], transpose=True)
        add(f"{pfx}_self_q:bias", sd[f"{lp}attention.self.query.bias"])
        add(f"{pfx}_self_k:weight", sd[f"{lp}attention.self.key.weight"], transpose=True)
        add(f"{pfx}_self_k:bias", sd[f"{lp}attention.self.key.bias"])
        add(f"{pfx}_self_v:weight", sd[f"{lp}attention.self.value.weight"], transpose=True)
        add(f"{pfx}_self_v:bias", sd[f"{lp}attention.self.value.bias"])

        add(f"{pfx}_self_out:weight", sd[f"{lp}attention.output.dense.weight"], transpose=True)
        add(f"{pfx}_self_out:bias", sd[f"{lp}attention.output.dense.bias"])
        add(f"{pfx}_self_ln:gamma", sd[f"{lp}attention.output.LayerNorm.weight"])
        add(f"{pfx}_self_ln:beta", sd[f"{lp}attention.output.LayerNorm.bias"])

        add(f"{pfx}_cross_q:weight", sd[f"{lp}crossattention.self.query.weight"], transpose=True)
        add(f"{pfx}_cross_q:bias", sd[f"{lp}crossattention.self.query.bias"])
        add(f"{pfx}_cross_k:weight", sd[f"{lp}crossattention.self.key.weight"], transpose=True)
        add(f"{pfx}_cross_k:bias", sd[f"{lp}crossattention.self.key.bias"])
        add(f"{pfx}_cross_v:weight", sd[f"{lp}crossattention.self.value.weight"], transpose=True)
        add(f"{pfx}_cross_v:bias", sd[f"{lp}crossattention.self.value.bias"])

        add(f"{pfx}_cross_out:weight", sd[f"{lp}crossattention.output.dense.weight"], transpose=True)
        add(f"{pfx}_cross_out:bias", sd[f"{lp}crossattention.output.dense.bias"])
        add(f"{pfx}_cross_ln:gamma", sd[f"{lp}crossattention.output.LayerNorm.weight"])
        add(f"{pfx}_cross_ln:beta", sd[f"{lp}crossattention.output.LayerNorm.bias"])

        add(f"{pfx}_ffn_inter:weight", sd[f"{lp}intermediate.dense.weight"], transpose=True)
        add(f"{pfx}_ffn_inter:bias", sd[f"{lp}intermediate.dense.bias"])
        add(f"{pfx}_ffn_out:weight", sd[f"{lp}output.dense.weight"], transpose=True)
        add(f"{pfx}_ffn_out:bias", sd[f"{lp}output.dense.bias"])
        add(f"{pfx}_ffn_ln:gamma", sd[f"{lp}output.LayerNorm.weight"])
        add(f"{pfx}_ffn_ln:beta", sd[f"{lp}output.LayerNorm.bias"])

    # 7. LM head (vocab projection is TIED - not re-saved)
    cp = "cls."
    add("lmhead_dense:weight", sd[f"{cp}predictions.transform.dense.weight"], transpose=True)
    add("lmhead_dense:bias", sd[f"{cp}predictions.transform.dense.bias"])
    add("lmhead_ln:gamma", sd[f"{cp}predictions.transform.LayerNorm.weight"])
    add("lmhead_ln:beta", sd[f"{cp}predictions.transform.LayerNorm.bias"])
    add("lm_head_bias/weights:lmhead_bias", sd[f"{cp}predictions.bias"])

    return weights


# ---------------------------------------------------------------------------
# Shape validation - a wrong-sized tensor silently produces garbage captions,
# so assert the v4.0.0 contract up front rather than at inference time.
# ---------------------------------------------------------------------------

def validate(enc_weights, dec_weights):
    """Assert the emitted tensors match the v4.0.0 architecture contract."""
    enc = dict(enc_weights)
    dec = dict(dec_weights)

    expected = {
        "patch_embed_conv:filter": (768, 3, 16, 16),
        "pos_embedding:pos_embedding": (1, 1, 576, 768),
        "enc_layer0_wq:weight": (768, 768),
        "enc_layer0_fc1:weight": (768, 3072),
        "post_ln:gamma": (768,),
        "enc_to_dec_proj:weight": (768, 512),
        "enc_to_dec_proj:bias": (512,),
    }
    for name, shape in expected.items():
        got = enc[name].shape
        assert got == shape, f"encoder {name}: expected {shape}, got {got}"

    expected = {
        "word_emb:Embedding": (30522, 512),
        "pos_emb:Embedding": (512, 512),
        "type_emb:Embedding": (2, 512),
        "dec_layer0_self_q:weight": (512, 512),
        # Cross K/V are 512x512, not 768x512: the connector projects the
        # encoder output to the decoder width before cross-attention sees it.
        "dec_layer0_cross_k:weight": (512, 512),
        "dec_layer0_ffn_inter:weight": (512, 2048),
        "lmhead_dense:weight": (512, 512),
        "lm_head_bias/weights:lmhead_bias": (30522,),
    }
    for name, shape in expected.items():
        got = dec[name].shape
        assert got == shape, f"decoder {name}: expected {shape}, got {got}"

    print("Shape validation: OK (v4.0.0 contract)")


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Convert the ScreenAI caption v4.0.0 checkpoint bundle "
        "(SigLIP2 encoder + BERT decoder + connector) to nntrainer format."
    )
    p.add_argument("--bundle", type=str, required=True,
                   help="Path to the extracted v4.0.0-S1 bundle root")
    p.add_argument("--encoder_output", type=str,
                   default="./nntr_siglip2_encoder_fp32.bin",
                   help="Output path for the encoder weight file")
    p.add_argument("--decoder_output", type=str,
                   default="./nntr_caption_decoder_fp32.bin",
                   help="Output path for the decoder weight file")
    p.add_argument("--target", type=str, default="both",
                   choices=["encoder", "decoder", "both"],
                   help="Which half to convert")
    p.add_argument("--data_type", type=str, default="float32",
                   choices=["float32"], help="Output data type")
    p.add_argument("--safetensors", action="store_true",
                   help="Save safetensors instead of raw binary")
    return p.parse_args()


def main():
    """Convert the v4.0.0 bundle to nntrainer weight files."""
    args = parse_args()
    bundle = Path(args.bundle)
    dtype = args.data_type

    enc_path = bundle / "siglip2-base-patch16-384" / "model.safetensors"
    dec_path = bundle / "best" / "decoder" / "model.safetensors"
    proj_path = bundle / "best" / "encoder_to_decoder.pt"
    for path in (enc_path, dec_path, proj_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing v4.0.0 artifact: {path}")

    print(f"Loading encoder: {enc_path}")
    enc_sd = load_file(str(enc_path))
    print(f"Loading decoder: {dec_path}")
    dec_sd = load_file(str(dec_path))
    print(f"Loading connector: {proj_path}")
    proj = torch.load(str(proj_path), map_location="cpu", weights_only=True)
    print(f"Loaded - encoder {len(enc_sd)}, decoder {len(dec_sd)}, "
          f"connector {len(proj)} tensors.")

    enc_weights = collect_encoder(enc_sd, proj, dtype)
    dec_weights = collect_decoder(dec_sd, dtype)
    print(f"\nEncoder tensors: {len(enc_weights)}")
    print(f"Decoder tensors: {len(dec_weights)}")
    assert len(enc_weights) == EXPECTED_ENC_TENSORS, (
        f"ENCODER COUNT MISMATCH: got {len(enc_weights)}, "
        f"expected {EXPECTED_ENC_TENSORS}")
    assert len(dec_weights) == EXPECTED_DEC_TENSORS, (
        f"DECODER COUNT MISMATCH: got {len(dec_weights)}, "
        f"expected {EXPECTED_DEC_TENSORS}")
    validate(enc_weights, dec_weights)

    if args.target in ("encoder", "both"):
        if args.safetensors:
            save_safetensors(enc_weights,
                             get_safetensors_output_name(args.encoder_output), dtype)
        else:
            save_bin(enc_weights, args.encoder_output)
    if args.target in ("decoder", "both"):
        if args.safetensors:
            save_safetensors(dec_weights,
                             get_safetensors_output_name(args.decoder_output), dtype)
        else:
            save_bin(dec_weights, args.decoder_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
