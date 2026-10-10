from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def _load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def compute_clip_score(prompt: str, image_path: str | Path) -> float:
    """Compute a CLIPScore-like similarity between prompt and image."""
    model, processor = _load_clip_model()
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
    score = torch.softmax(logits_per_image, dim=1)[0, 0].item()
    return float(score)


def compute_aesthetic_score(image_path: str | Path) -> float:
    """Placeholder aesthetic scoring using a simple image-based heuristic."""
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    brightness = float(arr.mean()) / 255.0
    contrast = float(arr.std()) / 255.0
    return float(0.5 * brightness + 0.5 * contrast)


def compute_metrics_for_records(records: list[dict]) -> list[dict]:
    enriched = []
    for item in records:
        prompt = item["prompt"]
        image_path = item["image_path"]
        clip_score = compute_clip_score(prompt, image_path)
        aesthetic_score = compute_aesthetic_score(image_path)
        enriched.append({
            **item,
            "clip_score": clip_score,
            "aesthetic_score": aesthetic_score,
        })
    return enriched


def save_metric_report(records: list[dict], output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
