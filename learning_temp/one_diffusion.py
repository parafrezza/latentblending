
import math
import torch
import numpy as np

from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Carica una pipeline SDXL pre-addestrata (UNet + scheduler + VAE)
def load_sdxl_pipe(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"Pipeline loaded on {device}.")
    return pipe

def normalize_size(height, width, multiple=8):
    new_height = height - (height % multiple)
    new_width = width - (width % multiple)
    if new_height != height or new_width != width:
        new_height = max(multiple, new_height)
        new_width = max(multiple, new_width)
        print(f"Adjusted size to {new_width}x{new_height} (divisible by {multiple}).")
    return new_height, new_width

# Crea gli embedding SDXL e i time ids necessari alla UNet
def get_sdxl_conditioning(pipe, prompt, negative_prompt=None, guidance_scale=0.0, height=None, width=None):
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    if do_cfg and negative_prompt is None:
        negative_prompt = ""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )

    if height is None or width is None:
        height = pipe.default_sample_size * pipe.vae_scale_factor
        width = pipe.default_sample_size * pipe.vae_scale_factor

    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    return prompt_embeds, pooled_prompt_embeds, add_time_ids, do_cfg
# Inizializza un vettore latente casuale
def initialize_latent_vector(pipe, seed=420, height=None, width=None, verbose=True):
    if height is None or width is None:
        height = pipe.default_sample_size * pipe.vae_scale_factor
        width = pipe.default_sample_size * pipe.vae_scale_factor
    generator = torch.Generator(device=pipe._execution_device).manual_seed(int(seed))
    latents = pipe.prepare_latents(
        1,
        pipe.unet.config.in_channels,
        height,
        width,
        pipe.unet.dtype,
        pipe._execution_device,
        generator,
        None,
    )
    if verbose:
        print("Initialized latent vector shape:", latents.shape)
    return latents

# Funzione di denoising usando UNet e scheduler pre-addestrati
def denoise(
    pipe,
    latent_vector,
    prompt_embeds,
    pooled_prompt_embeds,
    add_time_ids,
    num_steps=4,
    guidance_scale=0.0,
    do_cfg=False,
    save_intermediate=False,
    intermediate_every=1,
    include_start=True,
    verbose=True,
    analyze_intermediate=False,
    analysis_every=1,
):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=pipe._execution_device)
    latents = latent_vector
    intermediate_images = []
    if save_intermediate and include_start:
        if verbose:
            print("Saving step 0 (initial noise).")
        img = decode_from_latent(pipe, latents, verbose=False)
        intermediate_images.append(img)
        if analyze_intermediate:
            stats = analyze_image_stats(img)
            report_image_stats("Step 0", stats)
    for step, t in enumerate(scheduler.timesteps):
        if verbose:
            print(f"Denoising step {step+1}/{num_steps}")
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if verbose:
            print("Updated latent vector shape:", latents.shape)
        if save_intermediate and (step % intermediate_every == 0 or step == len(scheduler.timesteps) - 1):
            img = decode_from_latent(pipe, latents, verbose=False)
            intermediate_images.append(img)
            if analyze_intermediate and (step % analysis_every == 0 or step == len(scheduler.timesteps) - 1):
                stats = analyze_image_stats(img)
                report_image_stats(f"Step {step+1}", stats)
    if verbose:
        print("Denoising completed. Final latent vector shape:", latents.shape)
    if save_intermediate:
        return latents, intermediate_images
    return latents
# Decodifica il vettore latente in un'immagine
def decode_from_latent(pipe, latent_vector, verbose=True):
    if verbose:
        print("Decoding with VAE.")
    with torch.no_grad():
        decoded_image = pipe.vae.decode(
            latent_vector / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
    if verbose:
        print("Decoded image shape:", decoded_image.shape)
    decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2  # scale to [0, 1]
    decoded_image = decoded_image.squeeze().permute(1, 2, 0).cpu().numpy()  # HWC
    decoded_image = (decoded_image * 255).astype(np.uint8)
    img = Image.fromarray(decoded_image)
    if verbose:
        print("Image decoded from latent vector.")
    return img

def analyze_image_stats(
    img,
    low=8,
    high=247,
    dark_ratio_threshold=0.97,
    bright_ratio_threshold=0.97,
    flat_std_threshold=5.0,
):
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    mean = float(arr.mean())
    std = float(arr.std())
    low_ratio = float((arr <= low).mean())
    high_ratio = float((arr >= high).mean())
    flags = []
    if low_ratio >= dark_ratio_threshold:
        flags.append("mostly_dark")
    if high_ratio >= bright_ratio_threshold:
        flags.append("mostly_bright")
    if std <= flat_std_threshold:
        flags.append("low_contrast")
    return {
        "mean": mean,
        "std": std,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "flags": flags,
    }

def report_image_stats(label, stats):
    flags = stats["flags"]
    status = "ok" if not flags else ",".join(flags)
    print(
        f"{label}: mean={stats['mean']:.1f} std={stats['std']:.1f} "
        f"low={stats['low_ratio']:.0%} high={stats['high_ratio']:.0%} status={status}"
    )

def find_max_square_resolution(
    pipe,
    prompt,
    negative_prompt=None,
    guidance_scale=0.0,
    min_size=512,
    max_size=2048,
    step=128,
    num_steps=1,
    safety_margin=0.9,
):
    if not torch.cuda.is_available():
        print("CUDA non disponibile: impossibile stimare la risoluzione massima.")
        return None
    device = pipe._execution_device
    if isinstance(device, torch.device) and device.type != "cuda":
        print("Pipeline su CPU: impossibile stimare la risoluzione massima in modo affidabile.")
        return None
    device_index = device.index if isinstance(device, torch.device) and device.index is not None else 0
    total_mem = torch.cuda.get_device_properties(device_index).total_memory
    best = None
    for size in range(min_size, max_size + 1, step):
        height, width = normalize_size(size, size, multiple=8)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            prompt_embeds, pooled_prompt_embeds, add_time_ids, do_cfg = get_sdxl_conditioning(
                pipe,
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
            latents = initialize_latent_vector(
                pipe,
                seed=420,
                height=height,
                width=width,
                verbose=False,
            )
            _ = denoise(
                pipe,
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                add_time_ids,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                do_cfg=do_cfg,
                save_intermediate=False,
                verbose=False,
            )
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device_index)
            ratio = peak / total_mem if total_mem else 0.0
            print(f"Probe {width}x{height}: peak {peak/1e9:.2f}GB ({ratio:.0%} of total)")
            if ratio < safety_margin:
                best = (height, width)
            else:
                print("Stopping: raggiunto il margine di sicurezza.")
                break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM a {width}x{height}.")
            break
        finally:
            torch.cuda.empty_cache()
    return best

def build_image_grid(images, cols=4, padding=8, bg_color=(0, 0, 0)):
    if not images:
        return None
    img_w, img_h = images[0].size
    rows = int(math.ceil(len(images) / cols))
    grid_w = cols * img_w + (cols - 1) * padding
    grid_h = rows * img_h + (rows - 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=bg_color)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * (img_w + padding)
        y = row * (img_h + padding)
        grid.paste(img, (x, y))
    return grid
# Salva l'immagine risultante
def save_image(img, path="generated_image.png"):
    img.save(path)
    print(f"Image saved to {path}")


if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = "A beautiful landscape painting of mountains during sunset."
    negative_prompt = "blurry, low quality, artifacts"
    guidance_scale = 7.0
    num_steps = 25
    height = 512
    width  = 512
    save_intermediate = True
    intermediate_every = 1
    analyze_intermediate = True
    analysis_every = 1
    grid_cols = 4
    probe_max_resolution = False
    probe_min_size = 512
    probe_max_size = 1536
    probe_step = 128
    probe_steps = 1
    probe_safety_margin = 0.9

    pipe = load_sdxl_pipe(model_id=model_id)
    if probe_max_resolution:
        best = find_max_square_resolution(
            pipe,
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            min_size=probe_min_size,
            max_size=probe_max_size,
            step=probe_step,
            num_steps=probe_steps,
            safety_margin=probe_safety_margin,
        )
        if best is not None:
            height, width = best
            print(f"Using probed resolution: {width}x{height}")
    height, width = normalize_size(height, width, multiple=8)
    if height != 1024 or width != 1024:
        print("Nota: SDXL e' addestrato a 1024x1024; altre risoluzioni possono degradare la qualita'.")
    prompt_embeds, pooled_prompt_embeds, add_time_ids, do_cfg = get_sdxl_conditioning(
        pipe,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    print(f"CFG attivo: {do_cfg} (guidance_scale={guidance_scale})")
    latent_vector = initialize_latent_vector(pipe, seed=420, height=height, width=width)
    denoise_result = denoise(
        pipe,
        latent_vector,
        prompt_embeds,
        pooled_prompt_embeds,
        add_time_ids,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        do_cfg=do_cfg,
        save_intermediate=save_intermediate,
        intermediate_every=intermediate_every,
        analyze_intermediate=analyze_intermediate,
        analysis_every=analysis_every,
    )
    if save_intermediate:
        denoised_latent, intermediate_images = denoise_result
    else:
        denoised_latent = denoise_result
        intermediate_images = []
    decoded_image = decode_from_latent(pipe, denoised_latent)
    if analyze_intermediate:
        report_image_stats("Final", analyze_image_stats(decoded_image))
    save_image(decoded_image, "generated_image.png")
    if save_intermediate and intermediate_images:
        grid = build_image_grid(intermediate_images, cols=grid_cols)
        if grid is not None:
            save_image(grid, "denoise_grid.png")
