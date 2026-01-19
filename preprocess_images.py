#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple) * multiple))


def _load_and_prepare_image(path, size, multiple=32):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size

    if size is not None:
        target_w, target_h = size
    else:
        target_w = _round_to_multiple(orig_w, multiple)
        target_h = _round_to_multiple(orig_h, multiple)

    if (target_w, target_h) != (orig_w, orig_h):
        img = img.resize((target_w, target_h), resample=Image.LANCZOS)

    return img, (orig_w, orig_h), (target_w, target_h)


def _encode_latent(pipe, img):
    image = torch.from_numpy(np.array(img)).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image * 2.0 - 1.0

    needs_upcast = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcast:
        pipe.upcast_vae()
        image = image.to(dtype=torch.float32)
    else:
        image = image.to(dtype=pipe.vae.dtype)

    image = image.to(device=pipe._execution_device)
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    if needs_upcast:
        pipe.vae.to(dtype=torch.float16)

    return latents


def main():
    parser = argparse.ArgumentParser(description="Preprocess two images and save VAE latents.")
    parser.add_argument("--image1", required=True, help="Path to first image.")
    parser.add_argument("--image2", required=True, help="Path to second image.")
    parser.add_argument("--output-dir", default="preprocessed", help="Directory to store outputs.")
    parser.add_argument("--model", default="stabilityai/sdxl-turbo", help="Diffusers model id.")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Optional resize target (must be multiples of 32).")
    parser.add_argument("--ask-to-blend", action="store_true",
                        help="Prompt the user to proceed with blending after success.")
    args = parser.parse_args()

    img1_path = Path(args.image1)
    img2_path = Path(args.image2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img1_path.is_file():
        raise FileNotFoundError(f"image1 not found: {img1_path}")
    if not img2_path.is_file():
        raise FileNotFoundError(f"image2 not found: {img2_path}")

    size = None
    if args.size is not None:
        w, h = args.size
        if w % 32 != 0 or h % 32 != 0:
            raise ValueError("Provided --size must be multiples of 32.")
        size = (w, h)

    img1, img1_orig, img1_final = _load_and_prepare_image(img1_path, size)
    img2, img2_orig, img2_final = _load_and_prepare_image(img2_path, size)

    if min(img1_final + img2_final) < 32:
        raise ValueError("Images are too small after resizing; minimum size is 32x32.")

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    latent1 = _encode_latent(pipe, img1).detach().cpu()
    latent2 = _encode_latent(pipe, img2).detach().cpu()

    img1_out = out_dir / "image1_preprocessed.png"
    img2_out = out_dir / "image2_preprocessed.png"
    img1.save(img1_out)
    img2.save(img2_out)

    latent1_out = out_dir / "image1_latent.pt"
    latent2_out = out_dir / "image2_latent.pt"
    torch.save(latent1, latent1_out)
    torch.save(latent2, latent2_out)

    meta = {
        "model": args.model,
        "image1": {
            "input": str(img1_path),
            "output": str(img1_out),
            "orig_size": img1_orig,
            "final_size": img1_final,
            "latent_path": str(latent1_out),
        },
        "image2": {
            "input": str(img2_path),
            "output": str(img2_out),
            "orig_size": img2_orig,
            "final_size": img2_final,
            "latent_path": str(latent2_out),
        },
    }
    meta_path = out_dir / "preprocess_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Preprocessing completed.")
    print(f"- manifest: {meta_path}")
    print(f"- image1: {img1_out}")
    print(f"- image2: {img2_out}")
    print(f"- latent1: {latent1_out}")
    print(f"- latent2: {latent2_out}")

    if args.ask_to_blend:
        reply = input("Vuoi usare queste immagini convertite per avviare il blending? (y/N): ").strip().lower()
        if reply in {"y", "yes"}:
            print("Perfetto. Tieni a portata di mano il manifest:")
            print(f"  {meta_path}")
            print("Nel prossimo step possiamo collegare questi latenti al blending engine.")


if __name__ == "__main__":
    main()
