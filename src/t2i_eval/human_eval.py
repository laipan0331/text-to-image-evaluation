from __future__ import annotations

import json
from pathlib import Path


HUMAN_RUBRIC = {
    "instruction_following": {
        "description": "Does the image satisfy the key prompt constraints?",
        "score_0": "Major mismatch with the prompt",
        "score_1": "Minor mismatch or partially correct",
        "score_2": "Mostly correctly follows the prompt",
        "score_3": "Fully follows the prompt",
    },
    "composition": {
        "description": "Is the layout visually coherent and balanced?",
        "score_0": "Chaotic or broken composition",
        "score_1": "Some structure but awkward",
        "score_2": "Reasonably balanced",
        "score_3": "Strong, clean composition",
    },
    "detail_quality": {
        "description": "Are object details and rendering sharp enough?",
        "score_0": "Blurry or distorted details",
        "score_1": "Noticeable imperfections",
        "score_2": "Mostly good details",
        "score_3": "Excellent detail fidelity",
    },
    "overall_aesthetics": {
        "description": "Overall visual appeal and realism or intended style",
        "score_0": "Unpleasant or broken",
        "score_1": "Weak aesthetic quality",
        "score_2": "Good overall output",
        "score_3": "Very strong aesthetic quality",
    },
}


def build_human_rubric() -> dict:
    return HUMAN_RUBRIC


def compute_annotation_agreement(labels_a: list[int], labels_b: list[int]) -> float:
    """A simple agreement estimate for two annotators.
    For a real study, replace with Cohen's kappa or Krippendorff's alpha.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Annotator label lengths must match.")
    if not labels_a:
        return 0.0
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return matches / len(labels_a)


def save_human_labels(records: list[dict], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
