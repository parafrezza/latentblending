#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from diffusers import AutoPipelineForText2Image

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


if __name__ == "__main__":
    main()
