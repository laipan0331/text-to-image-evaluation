from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


def load_model_pipeline(model_name: str, device: str | None = None, torch_dtype: torch.dtype | None = None):
    """Load a diffusion model from Hugging Face using diffusers."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)

    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    return pipe


def generate_images(prompt_bank: list[dict], model_name: str, output_dir: str | Path, num_per_prompt: int = 2, seed_offset: int = 0):
    """Generate a small set of images per prompt and save metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_model_pipeline(model_name)
    result_rows: list[dict] = []

    for index, item in enumerate(prompt_bank):
        prompt = item["prompt"]
        for image_idx in range(num_per_prompt):
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            seed = seed_offset + index * num_per_prompt + image_idx
            generator.manual_seed(seed)
            image = pipe(prompt=prompt, generator=generator, num_inference_steps=25).images[0]
            image_name = f"{item['prompt_id']:04d}_{image_idx:02d}_{seed}.png"
            image_path = output_dir / image_name
            image.save(image_path)

            record = dict(item)
            record["model_name"] = model_name
            record["seed"] = seed
            record["image_path"] = str(image_path)
            result_rows.append(record)

    metadata_path = output_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(result_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_rows
