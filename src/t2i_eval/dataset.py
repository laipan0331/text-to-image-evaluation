from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PromptEntry:
    prompt_id: int
    category: str
    prompt: str
    model_name: str = ""
    seed: int = 0
    image_path: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CATEGORY_TEMPLATES = {
    "object_count": [
        "three red apples on a wooden table",
        "five books stacked neatly beside a lamp",
        "two cats sitting on a windowsill, photorealistic",
        "seven balloons floating above a city street at sunset",
    ],
    "spatial_relation": [
        "a dog to the left of a bicycle in a park",
        "a cup on top of a notebook next to a laptop",
        "a small plant in front of a blue chair inside a studio",
        "a teddy bear behind a vase on a dining table",
    ],
    "color_attribute": [
        "a yellow lemon next to a green cucumber on a white counter",
        "a purple umbrella with a red handle under a cloudy sky",
        "a blue ceramic mug with gold trim on a wooden desk",
        "a black-and-white striped cat staring at a camera",
    ],
    "text_rendering": [
        "a street sign that says 'WELCOME HOME' in bold modern type",
        "a coffee cup with the text 'KEEP GOING' printed on it",
        "a book cover reading 'MIRROR CITY' with neon letters",
        "a chalkboard with the phrase 'AI LAB' written in neat handwriting",
    ],
    "style_composition": [
        "cinematic portrait of a chef in a warm kitchen, shallow depth of field",
        "watercolor painting of a mountain lake at sunrise, soft brush strokes",
        "isometric 3D office workspace with large glass windows, minimalistic design",
        "anime-style city alley at night with neon reflections and rainy pavement",
    ],
    "attribute_binding": [
        "a red bicycle with a blue basket beside a white wall",
        "a striped green shirt on a person holding a yellow umbrella",
        "a silver teapot with a black handle and a white flower decoration",
        "a pink sofa with a brown wooden frame in a cozy room",
    ],
}


def build_prompt_bank(num_per_category: int = 20, output_path: str | None = None) -> list[dict[str, Any]]:
    """Generate a prompt bank with balanced category coverage."""
    prompts: list[dict[str, Any]] = []
    for category, templates in CATEGORY_TEMPLATES.items():
        for idx in range(num_per_category):
            base = templates[idx % len(templates)]
            suffix = f" {category} benchmark sample {idx + 1}"
            prompt_text = f"{base}{suffix}" if idx % 3 == 0 else base
            prompts.append(
                {
                    "prompt_id": len(prompts) + 1,
                    "category": category,
                    "prompt": prompt_text,
                    "model_name": "",
                    "seed": 0,
                    "image_path": "",
                    "notes": "seeded benchmark prompt",
                }
            )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")

    return prompts


def load_prompt_bank(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def save_prompt_bank(path: str | Path, prompts: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(prompts), indent=2, ensure_ascii=False), encoding='utf-8')
