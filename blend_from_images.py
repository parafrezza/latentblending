#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import AutoPipelineForText2Image, StableDiffusionLatentUpscalePipeline

from latentblending.diffusers_holder import DiffusersHolder
from latentblending.utils import interpolate_spherical, interpolate_linear


def _load_latent(path, device):
    latent = torch.load(path, map_location="cpu")
    if not torch.is_tensor(latent):
        raise ValueError(f"Latent at {path} is not a torch tensor.")
    return latent.to(device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Blend two preprocessed image latents into a transition."
    )
    parser.add_argument(
        "--manifest",
        default="preprocessed/preprocess_manifest.json",
        help="Path to preprocess manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="blend_from_images",
        help="Directory to store blended frames.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames in the transition (including endpoints).",
    )
    parser.add_argument(
        "--interpolation",
        choices=["slerp", "linear"],
        default="slerp",
        help="Interpolation type in latent space.",
    )
    parser.add_argument(
        "--mp4",
        default=None,
        help="Optional path to write an MP4 movie.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the MP4 output.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model id to override the manifest model.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply a light denoising pass (fastNlMeansDenoisingColored) on each frame.",
    )
    parser.add_argument(
        "--denoise-h",
        type=int,
        default=7,
        help="Denoising strength for luminance/chrominance (OpenCV h parameter).",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Run a latent upscaling pass (2x) on the generated frames.",
    )
    parser.add_argument(
        "--upscale-model",
        default="stabilityai/sd-x2-latent-upscaler",
        help="Upscaler model id.",
    )
    parser.add_argument(
        "--upscale-steps",
        type=int,
        default=20,
        help="Number of inference steps for the upscaler.",
    )
    parser.add_argument(
        "--upscale-guidance",
        type=float,
        default=0.0,
        help="Guidance scale for the upscaler (typically 0 for fidelity).",
    )
    parser.add_argument(
        "--upscale-prompt",
        default="high quality, detailed",
        help="Prompt used during upscaling (can stay generic).",
    )
    parser.add_argument(
        "--mp4-upscaled",
        default=None,
        help="Optional path to write an MP4 movie of upscaled frames (requires --upscale).",
    )
    args = parser.parse_args()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)
    print("pipe device:", pipe.device, "vae dtype:", pipe.vae.dtype, "unet dtype:", pipe.unet.dtype)
    dh = DiffusersHolder(pipe)

    latent1 = _load_latent(latent1_path, device)
    latent2 = _load_latent(latent2_path, device)
    if latent1.shape != latent2.shape:
        raise ValueError(
            f"Latent shapes do not match: {latent1.shape} vs {latent2.shape}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fracts = torch.linspace(0, 1, args.num_frames)
    writer = None
    t_start = time.time()
    for i, f in enumerate(fracts):
        if args.interpolation == "slerp":
            latent = interpolate_spherical(latent1, latent2, float(f))
        else:
            latent = interpolate_linear(latent1, latent2, float(f))
        img = dh.latent2image(latent)

        if args.denoise:
            arr = np.array(img)
            arr = cv2.fastNlMeansDenoisingColored(
                arr,
                None,
                h=args.denoise_h,
                hColor=args.denoise_h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            img = Image.fromarray(arr)
        img.save(out_dir / f"frame_{i:04d}.png")
        if args.mp4 is not None:
            if writer is None:
                w, h = img.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.mp4), fourcc, args.fps, (w, h))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)

        # Progress with ETA
        elapsed = time.time() - t_start
        done = i + 1
        eta = (args.num_frames - done) * (elapsed / done)
        pct = (done / args.num_frames) * 100
        print(
            f"[blend_images] {done}/{args.num_frames} ({pct:.1f}%) elapsed {elapsed:.1f}s ETA {eta:.1f}s",
            end="\r",
            flush=True,
        )

    if writer is not None:
        writer.release()

    # Ensure the progress line ends nicely
    print()

    if args.upscale:
        up_dir = out_dir / "upscaled"
        up_dir.mkdir(parents=True, exist_ok=True)

        up_pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            args.upscale_model,
            torch_dtype=torch.float16,
        ).to(device)

        up_writer = None
        if args.mp4_upscaled is not None:
            first = Image.open(out_dir / "frame_0000.png")
            up_first = up_pipe(
                prompt=args.upscale_prompt,
                image=first,
                num_inference_steps=args.upscale_steps,
                guidance_scale=args.upscale_guidance,
            ).images[0]
            w, h = up_first.size
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            up_writer = cv2.VideoWriter(str(args.mp4_upscaled), fourcc, args.fps, (w, h))
            up_first.save(up_dir / "frame_0000.png")
            up_writer.write(cv2.cvtColor(np.array(up_first), cv2.COLOR_RGB2BGR))
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, args.num_frames):
            img = Image.open(out_dir / f"frame_{i:04d}.png")
            up_img = up_pipe(
                prompt=args.upscale_prompt,
                image=img,
                num_inference_steps=args.upscale_steps,
                guidance_scale=args.upscale_guidance,
            ).images[0]
            up_img.save(up_dir / f"frame_{i:04d}.png")
            if up_writer is not None:
                up_writer.write(cv2.cvtColor(np.array(up_img), cv2.COLOR_RGB2BGR))

        if up_writer is not None:
            up_writer.release()

        print("Upscale completed.")
        print(f"- upscaled frames: {up_dir}")
        if args.mp4_upscaled is not None:
            print(f"- upscaled mp4: {args.mp4_upscaled}")

    print("Blend completed.")
    print(f"- frames: {args.num_frames}")
    print(f"- output: {out_dir}")
    if args.mp4 is not None:
        print(f"- mp4: {args.mp4}")


if __name__ == "__main__":
    main()

# esegui con il comando:
# python blend_from_images.py --manifest t ./output_folder/preprocess_manifest.json --output-dir blend_output --num-frames 20 --interpolation slerp --mp4 blend_output/transition.mp4 --fps 30
# python blend_from_images.py --manifest ./output_folder/preprocess_manifest.json --output-dir blend_output --num-frames 120 --interpolation slerp --mp4 blend_output/nina_02.mp4 --mp4-upscaled nina_02_upscaled.mp4- -fps 30 --upscale --denoise