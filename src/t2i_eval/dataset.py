"""Benchmark prompt bank with element checklists (paper Sec. 4, contribution 1).

Five categories, as in the paper: counting, spatial, color_binding,
text_rendering, compound. Every prompt carries a structured checklist
({objects, relations, text, style}) that the judge verifies element by element.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

CATEGORIES = ["counting", "spatial", "color_binding", "text_rendering", "compound"]

OBJECTS = ["apple", "cat", "book", "balloon", "cup", "chair", "bird", "candle",
           "lantern", "bicycle", "dog", "hat", "bottle", "vase", "flower", "lamp", "kite"]
COLORS = ["red", "blue", "green", "yellow", "purple", "pink", "white", "black"]
CONTEXTS = ["on a wooden table", "in a park", "on a white background"]
RELATIONS = ["to the left of", "to the right of", "on top of", "behind"]
WORDS = ["SALE", "OPEN", "WELCOME", "KEEP GOING", "MIRROR CITY", "AI LAB", "COFFEE",
         "HELLO", "EXIT", "NEW", "FRESH", "HOME", "PEACE", "MENU", "STOP", "JAZZ"]
CARRIERS = ["a street sign", "a coffee cup", "a book cover", "a chalkboard", "a shop window",
            "a t-shirt", "a poster", "a neon sign"]
STYLES = ["flat poster style", "watercolor painting", "photorealistic", "isometric 3D render",
          "anime style", "pencil sketch"]


def _a(phrase: str) -> str:
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def _np(name: str, count: int, color: str | None = None) -> str:
    col = f"{color} " if color else ""
    return _a(col + name) if count == 1 else f"{count} {col}{name}s"


def _obj(name, count=1, color=None) -> dict:
    return {"name": name, "count": count, "color": color}


def _entry(category: str, objects=(), relations=(), text=(), style="") -> dict[str, Any]:
    return {"category": category,
            "checklist": {"objects": list(objects), "relations": list(relations),
                          "text": list(text), "style": style}}


def _counting(rng: random.Random):
    name, n, ctx = rng.choice(OBJECTS), rng.randint(2, 7), rng.choice(CONTEXTS)
    e = _entry("counting", [_obj(name, n)])
    e["prompt"] = f"{n} {name}s {ctx}"
    return e


def _spatial(rng: random.Random):
    a, b = rng.sample(OBJECTS, 2)
    rel = rng.choice(RELATIONS)
    e = _entry("spatial", [_obj(a), _obj(b)], [{"subject": a, "relation": rel, "object": b}])
    e["prompt"] = f"{_a(a)} {rel} {_a(b)}"
    return e


def _color_binding(rng: random.Random):
    a, b = rng.sample(OBJECTS, 2)
    ca, cb = rng.sample(COLORS, 2)
    e = _entry("color_binding", [_obj(a, 1, ca), _obj(b, 1, cb)])
    e["prompt"] = f"{_a(ca + ' ' + a)} and {_a(cb + ' ' + b)}"
    return e


def _text(rng: random.Random):
    word, carrier = rng.choice(WORDS), rng.choice(CARRIERS)
    e = _entry("text_rendering", [], [], [word])
    e["prompt"] = f"{carrier} with the text '{word}'"
    return e


def _compound(rng: random.Random):
    a, b = rng.sample(OBJECTS, 2)
    n, ca = rng.randint(2, 4), rng.choice(COLORS)
    rel, style = rng.choice(RELATIONS), rng.choice(STYLES)
    text = [rng.choice(WORDS)] if rng.random() < 0.5 else []
    e = _entry("compound", [_obj(a, n, ca), _obj(b)],
               [{"subject": a, "relation": rel, "object": b}], text, style)
    e["prompt"] = (f"{_np(a, n, ca)} {rel} {_a(b)}, {style}"
                   + (f", with the word '{text[0]}'" if text else ""))
    return e


_BUILDERS = {"counting": _counting, "spatial": _spatial, "color_binding": _color_binding,
             "text_rendering": _text, "compound": _compound}


def flatten_checklist(checklist: dict) -> list[dict]:
    """Turn the JSON checklist into atomic, individually verifiable elements."""
    out: list[dict] = []
    for o in checklist.get("objects", []):
        n = o.get("count", 1)
        if n != 1:
            out.append({"kind": "count", "count": n, "text": f"exactly {n} {o['name']}(s) are visible"})
        elif not o.get("color"):
            out.append({"kind": "object", "text": f"a {o['name']} is present"})
        if o.get("color"):
            out.append({"kind": "color", "text": (f"the {o['name']} is {o['color']}" if n == 1
                                          else f"the {o['name']}s are {o['color']}")})
    for r in checklist.get("relations", []):
        out.append({"kind": "relation", "text": f"the {r['subject']} is {r['relation']} the {r['object']}"})
    for t in checklist.get("text", []):
        out.append({"kind": "text", "text": f"the text '{t}' is rendered legibly and spelled correctly"})
    if checklist.get("style"):
        out.append({"kind": "style", "text": f"the image is in {checklist['style']}"})
    for i, el in enumerate(out):
        el["id"] = i
    return out


def build_prompt_bank(num_per_category: int = 30, output_path: str | None = None,
                      seed: int = 1234) -> list[dict[str, Any]]:
    """Deterministic, de-duplicated bank (default 5 x 30 = 150 prompts, within the paper's 100-200)."""
    rng = random.Random(seed)
    bank: list[dict[str, Any]] = []
    for category in CATEGORIES:
        seen: set[str] = set()
        attempts = 0
        while len(seen) < num_per_category and attempts < 10000:
            attempts += 1
            e = _BUILDERS[category](rng)
            if e["prompt"] in seen:
                continue
            seen.add(e["prompt"])
            e["prompt_id"] = len(bank) + 1
            e["elements"] = flatten_checklist(e["checklist"])
            e["seed"] = seed + e["prompt_id"]  # fixed per prompt, shared across models
            bank.append(e)
    if output_path is not None:
        save_json(output_path, bank)
    return bank


def load_json(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False), encoding="utf-8")
