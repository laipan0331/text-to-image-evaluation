"""Automatic metrics: CLIPScore and a lightweight aesthetic proxy (paper Sec. 4, contribution 4)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _load_clip():
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)


def compute_clip_score(prompt: str, image_path: str | Path) -> float:
    """CLIPScore (Hessel et al., 2021): 2.5 * max(cos(image, text), 0)."""
    import torch

    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        img = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    cos = torch.nn.functional.cosine_similarity(img, txt).item()
    return float(2.5 * max(cos, 0.0))


def compute_aesthetic_score(image_path: str | Path) -> float:
    """Heuristic proxy (NOT a learned aesthetic model): rewards mid-range brightness,
    contrast and colorfulness. Swap in the LAION aesthetic predictor for final numbers."""
    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    brightness = 1.0 - abs(float(arr.mean()) - 0.5) * 2
    contrast = min(float(arr.std()) / 0.25, 1.0)
    rg, yb = arr[..., 0] - arr[..., 1], 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorful = min(float(np.hypot(rg.std(), yb.std())) / 0.3, 1.0)
    return float((brightness + contrast + colorful) / 3)


def compute_metrics_for_records(records: list[dict]) -> list[dict]:
    return [{**r,
             "clip_score": compute_clip_score(r["prompt"], r["image_path"]),
             "aesthetic_score": compute_aesthetic_score(r["image_path"])} for r in records]
