from __future__ import annotations

import json
from pathlib import Path


def build_judge_prompt(prompt: str, image_path: str) -> str:
    return f"""
You are evaluating a text-to-image result.

Prompt: {prompt}
Image path: {image_path}

Return JSON with:
{{
  "instruction_following": 0.0,
  "object_count_match": 0.0,
  "color_match": 0.0,
  "spatial_match": 0.0,
  "text_match": 0.0,
  "overall_quality": 0.0,
  "reason": "short explanation"
}}

Score each field in [0, 1].
    """.strip()


def judge_with_vlm(prompt: str, image_path: str, model_client=None) -> dict:
    """Use a multimodal model as judge. This function is intentionally lightweight and works with an API client.
    If no client is supplied, it returns a heuristic JSON score.
    """
    if model_client is None:
        return {
            "instruction_following": 0.75,
            "object_count_match": 0.7,
            "color_match": 0.8,
            "spatial_match": 0.72,
            "text_match": 0.5,
            "overall_quality": 0.73,
            "reason": "fallback heuristic judge; replace with real VLM API for production benchmarking",
        }

    response = model_client.generate(
        prompt=build_judge_prompt(prompt, image_path),
        image_path=image_path,
    )
    return json.loads(response)


def process_judge_batch(records: list[dict], model_client=None) -> list[dict]:
    scored = []
    for item in records:
        verdict = judge_with_vlm(item["prompt"], item["image_path"], model_client)
        scored.append({**item, **verdict})
    return scored
