# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>

## @file run_pytorch_caption.py
## @brief PyTorch golden reference for the ScreenAI caption v4.0.0 bundle.
##
## Greedy-captions an image with the original PyTorch model so the nntrainer
## port can be diffed against it token-by-token. Optionally dumps the
## post-connector encoder hidden states (.npy) for numeric layer-level parity.
##
## The v4.0.0 model is assembled from three artifacts rather than a single
## VisionEncoderDecoder checkpoint, so this mirrors PELangCaptionModel.encode():
##   SiglipVisionModel(pixels).last_hidden_state  -> [1,576,768]
##   encoder_to_decoder(...)                      -> [1,576,512]
##   BertLMHeadModel(cross_attn over those states)
## @author Seungbaek Hong <sb92.hong@samsung.com>

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, BertLMHeadModel, SiglipVisionModel

IMG_SIZE = 384
RESAMPLE = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}


def preprocess(image_path, size, resample):
    """Resize to size*size and normalize with mean/std 0.5 (SigLIP2 recipe)."""
    img = Image.open(image_path).convert("RGB").resize((size, size), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)[None]  # [1,3,H,W] CHW
    return torch.from_numpy(np.ascontiguousarray(arr))


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="extracted v4.0.0-S1 root")
    p.add_argument("--image", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--resample", default="bicubic", choices=sorted(RESAMPLE))
    p.add_argument("--dump-encoder", type=str, default=None,
                   help="write post-connector encoder states to this .npy")
    p.add_argument("--dump-pixels", type=str, default=None,
                   help="write the preprocessed [1,3,H,W] pixel tensor to .npy")
    return p.parse_args()


def main():
    """Run greedy captioning and print token ids + decoded text."""
    args = parse_args()
    bundle = Path(args.bundle)

    encoder = SiglipVisionModel.from_pretrained(
        bundle / "siglip2-base-patch16-384", attn_implementation="eager").eval()
    decoder = BertLMHeadModel.from_pretrained(
        bundle / "best" / "decoder", attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(bundle / "best" / "tokenizer")

    proj_sd = torch.load(bundle / "best" / "encoder_to_decoder.pt",
                         map_location="cpu", weights_only=True)
    connect = torch.nn.Linear(768, 512)
    connect.load_state_dict(proj_sd)
    connect.eval()

    pixel = preprocess(args.image, IMG_SIZE, RESAMPLE[args.resample])
    if args.dump_pixels:
        np.save(args.dump_pixels, pixel.numpy())

    with torch.no_grad():
        feats = encoder(pixel_values=pixel, return_dict=True).last_hidden_state
        assert tuple(feats.shape) == (1, 576, 768), tuple(feats.shape)
        enc_states = connect(feats)  # [1,576,512]
        if args.dump_encoder:
            np.save(args.dump_encoder, enc_states.numpy())

        # Greedy decode. use_cache=False mirrors the reference generate() path.
        ids = torch.tensor([[tok.cls_token_id]], dtype=torch.long)
        for _ in range(args.max_new_tokens):
            logits = decoder(
                input_ids=ids,
                encoder_hidden_states=enc_states,
                use_cache=False,
            ).logits
            nxt = int(logits[0, -1].argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == tok.sep_token_id:
                break

    out = ids[0].tolist()
    print("image     :", args.image, f"({args.resample})")
    print("token_ids :", " ".join(map(str, out)))
    print("caption   :", tok.decode(out, skip_special_tokens=True))
    print("enc_stats : mean=%.6f std=%.6f min=%.6f max=%.6f" % (
        enc_states.mean(), enc_states.std(), enc_states.min(), enc_states.max()))


if __name__ == "__main__":
    main()
