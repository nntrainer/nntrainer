# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file verify_parity.py
## @brief PyTorch reference + nntrainer token comparison for caption model
import argparse, json
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel


def preprocess(image_path, size=224):
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                     # SigLIP normalization
    arr = arr.transpose(2, 0, 1)[None]          # [1,3,224,224]
    return torch.from_numpy(np.ascontiguousarray(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="golden.json")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--nntr-tokens", help="JSON file with nntrainer token_ids to compare")
    args = ap.parse_args()

    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    pixel = preprocess(args.image)

    with torch.no_grad():
        gen = model.generate(pixel_values=pixel, max_new_tokens=args.max_new_tokens,
                             num_beams=1, do_sample=False)
    token_ids = gen[0].tolist()
    caption = tok.decode(token_ids, skip_special_tokens=True)
    payload = {"token_ids": token_ids, "caption": caption}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("golden token_ids:", token_ids)
    print("golden caption:", caption)

    if args.dump:
        with torch.no_grad():
            enc = model.get_encoder()(pixel_values=pixel, return_dict=True).last_hidden_state
            proj = model.enc_to_dec_proj(enc) if getattr(model, "enc_to_dec_proj", None) else enc
            np.save(args.out + ".encoder_hidden.npy", proj.cpu().numpy())
            # Save the exact preprocessed pixel tensor so C++ can bypass its own
            # resize and consume identical pixels, isolating encoder math parity.
            np.save(args.out + ".pixel.npy", pixel.cpu().numpy())
            print("Saved pixel tensor to", args.out + ".pixel.npy",
                  "shape:", pixel.shape)
            start = torch.tensor([[model.config.decoder_start_token_id]])
            dout = model.decoder(input_ids=start, encoder_hidden_states=proj,
                                 use_cache=True, return_dict=True)
            np.save(args.out + ".decoder_init_logits.npy", dout.logits.cpu().numpy())

    if args.nntr_tokens:
        with open(args.nntr_tokens) as f:
            nntr = json.load(f)["token_ids"]
        match = nntr == token_ids
        print("TOKEN MATCH:", match)
        if not match:
            print("  golden:", token_ids)
            print("  nntr  :", nntr)
        raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
