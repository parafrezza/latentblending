#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
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
        # Preserve aspect ratio: scale to cover target, then center-crop to target size.
        scale = max(target_w / orig_w, target_h / orig_h)
        resized_w = max(target_w, int(math.ceil(orig_w * scale)))
        resized_h = max(target_h, int(math.ceil(orig_h * scale)))

        img = img.resize((resized_w, resized_h), resample=Image.LANCZOS)

        if (resized_w, resized_h) != (target_w, target_h):
            left = (resized_w - target_w) // 2
            top = (resized_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            img = img.crop((left, top, right, bottom))

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
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0", help="Diffusers model id (use turbo for speed: stabilityai/sdxl-turbo).")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Optional resize target (must be multiples of 32).")
    parser.add_argument("--ask-to-blend", action="store_true",
                        help="Prompt the user to proceed with blending after success.")
    args = parser.parse_args()

# launch with: python preprocess_images.py --image1 path/to/img1.png --image2 path/to/img2.png --output-dir output_folder --size 512 512 --ask-to-blend
# launch with:  python preprocess_images.py --image1 ./source/lr2.jpg --image2 ./source/lr1.jpg --output-dir output_folder --size 512 512 --ask-to-blend

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
        reply = input("Vuoi avviare subito il blending con questi latenti? (y/N): ").strip().lower()
        if reply in {"y", "yes"}:
            blend_script = Path(__file__).parent / "blend_from_images.py"
            cmd = [
                sys.executable,
                str(blend_script),
                "--manifest",
                str(meta_path),
            ]
            print("Lancio:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print("Blending terminato con errore:", exc)
            else:
                print("Blending completato.")


if __name__ == "__main__":
    main()


# nuovo venv con python 3.12
# py -3.12 -m venv venv
# .\venv\Scripts\python.exe -m pip install --upgrade pip
# .\venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# .\venv\Scripts\python.exe -m pip install -r requirements.txt

# 