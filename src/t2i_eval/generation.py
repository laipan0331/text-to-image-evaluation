"""Image generation for the two benchmarked models (paper: SDXL + SDXL-Turbo)."""
from __future__ import annotations

import time
from pathlib import Path

MODELS = {
    "sdxl": {"hf_id": "stabilityai/stable-diffusion-xl-base-1.0", "steps": 30, "guidance": 7.0},
    "sdxl-turbo": {"hf_id": "stabilityai/sdxl-turbo", "steps": 4, "guidance": 0.0},
}


def load_model_pipeline(hf_id: str, device: str | None = None):
    import torch
    from diffusers import AutoPipelineForText2Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoPipelineForText2Image.from_pretrained(hf_id, torch_dtype=dtype).to(device), device


def generate_images(bank: list[dict], model_key: str, output_dir: str | Path) -> list[dict]:
    """One image per prompt using the prompt's fixed seed, so both models see identical seeds."""
    import torch

    cfg = MODELS[model_key]
    out = Path(output_dir) / model_key
    out.mkdir(parents=True, exist_ok=True)
    pipe = device = None  # loaded lazily so a fully cached run needs no GPU
    rows = []
    for item in bank:
        path = out / f"{item['prompt_id']:04d}.png"
        latency = None
        if not path.exists():  # resume: keep images from an interrupted run
            if pipe is None:
                pipe, device = load_model_pipeline(cfg["hf_id"])
            gen = torch.Generator(device=device).manual_seed(item["seed"])
            t0 = time.time()
            img = pipe(prompt=item["prompt"], generator=gen, num_inference_steps=cfg["steps"],
                       guidance_scale=cfg["guidance"]).images[0]
            latency = round(time.time() - t0, 3)
            img.save(path)
        rows.append({**item, "model": model_key, "image_path": str(path),
                     "steps": cfg["steps"], "guidance": cfg["guidance"], "latency_s": latency})
    return rows
