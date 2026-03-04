#!/usr/bin/env python3
# TODO: research a macOS-compatible pipeline for diffusers (xref user request)
# TODO: consider a future mode that reads a folder of images, sorts them
#       alphabetically, and blends them in sequence with intermediate steps.
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
import cv2

from latentblending.diffusers_holder import DiffusersHolder
from latentblending.utils import interpolate_spherical, interpolate_linear
from latentblending.utils import get_device


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

def _load_latent(path, device):
    latent = torch.load(path, map_location="cpu")
    if not torch.is_tensor(latent):
        raise ValueError(f"Latent at {path} is not a torch tensor.")
    return latent.to(device=device)


def cmd_blend(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    model_id = args.model or manifest.get("model", "stabilityai/sdxl-turbo")
    latent1_path = Path(manifest["image1"]["latent_path"])
    latent2_path = Path(manifest["image2"]["latent_path"])
    if not latent1_path.is_file():
        raise FileNotFoundError(f"latent1 not found: {latent1_path}")
    if not latent2_path.is_file():
        raise FileNotFoundError(f"latent2 not found: {latent2_path}")

    device = get_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16"
    )
    pipe.to(device)
    dh = DiffusersHolder(pipe)

    latent1 = _load_latent(latent1_path, device)
    latent2 = _load_latent(latent2_path, device)
    if latent1.shape != latent2.shape:
        raise ValueError(f"Latent shapes do not match: {latent1.shape} vs {latent2.shape}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fracts = torch.linspace(0, 1, args.num_frames)
    writer = None
    for i, f in enumerate(fracts):
        if args.interpolation == "slerp":
            latent = interpolate_spherical(latent1, latent2, float(f))
        else:
            latent = interpolate_linear(latent1, latent2, float(f))
        img = dh.latent2image(latent)
        img.save(out_dir / f"frame_{i:04d}.png")
        if args.mp4 is not None:
            if writer is None:
                w, h = img.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.mp4), fourcc, args.fps, (w, h))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)

    if writer is not None:
        writer.release()

    print("Blend completed.")
    print(f"- frames: {args.num_frames}")
    print(f"- output: {out_dir}")
    if args.mp4 is not None:
        print(f"- mp4: {args.mp4}")


def cmd_preprocess(args):
    # previous preprocessing logic moved here
    # (args already holds the necessary attributes)
    img1_path = Path(args.image1)  # this function will be used by main directly
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

    device = get_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16"
    )
    pipe.to(device)

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


def main():
    # backward-compatible alias (historical invocation style)
    if "-all" in sys.argv:
        sys.argv[sys.argv.index("-all")] = "all"

    parser = argparse.ArgumentParser(description="Preprocess/blend images.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pre = sub.add_parser("preprocess", help="Encode two images to latents.")
    pre.add_argument("--image1", required=True)
    pre.add_argument("--image2", required=True)
    pre.add_argument("--output-dir", default="preprocessed")
    pre.add_argument("--model", default="stabilityai/sdxl-turbo")
    pre.add_argument("--size", nargs=2, type=int)
    pre.add_argument("--ask-to-blend", action="store_true")
    pre.set_defaults(func=cmd_preprocess)
    bl = sub.add_parser("blend", help="Blend using a manifest.")
    bl.add_argument("--manifest", default="preprocessed/preprocess_manifest.json")
    bl.add_argument("--output-dir", default="blend_from_images")
    bl.add_argument("--num-frames", type=int, default=16)
    bl.add_argument("--interpolation", choices=["slerp","linear"], default="slerp")
    bl.add_argument("--mp4")
    bl.add_argument("--fps", type=int, default=30)
    bl.add_argument("--model")
    bl.set_defaults(func=cmd_blend)
    allp = sub.add_parser("all", help="preprocess then blend")
    allp.add_argument("--image1", required=True)
    allp.add_argument("--image2", required=True)
    allp.add_argument("--output-dir", default="preprocessed")
    allp.add_argument("--model", default="stabilityai/sdxl-turbo")
    allp.add_argument("--size", nargs=2, type=int)
    allp.add_argument("--num-frames", type=int, default=16)
    allp.add_argument("--interpolation", choices=["slerp","linear"], default="slerp")
    allp.add_argument("--mp4")
    allp.add_argument("--fps", type=int, default=30)
    allp.set_defaults(func=cmd_all)
    args = parser.parse_args()
    args.func(args)


def cmd_all(args):
    # run both steps in sequence
    cmd_preprocess(args)
    manifest_path = Path(args.output_dir) / "preprocess_manifest.json"
    blend_args = argparse.Namespace(
        manifest=str(manifest_path),
        output_dir=(Path(args.output_dir) / "blend_from_images").as_posix(),
        num_frames=args.num_frames,
        interpolation=args.interpolation,
        mp4=args.mp4,
        fps=args.fps,
        model=args.model,
    )
    cmd_blend(blend_args)


if __name__ == "__main__":
    main()


# nuovo venv con python 3.12
# py -3.12 -m venv venv
# .\venv\Scripts\python.exe -m pip install --upgrade pip
# .\venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# .\venv\Scripts\python.exe -m pip install -r requirements.txt

# 