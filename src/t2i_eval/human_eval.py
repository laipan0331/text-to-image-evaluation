"""Blind, randomized human evaluation with agreement statistics (paper Sec. 4, contribution 3)."""
from __future__ import annotations

import csv
import random
from itertools import combinations
from pathlib import Path

import numpy as np

ANCHORS = {1: "unusable", 2: "poor", 3: "acceptable", 4: "good", 5: "excellent"}

HUMAN_RUBRIC = {
    "instruction_following": "How fully does the image satisfy every part of the request (counts, colors, relations, text, style)?",
    "composition": "Is the layout balanced, with a clear focal point and visual hierarchy?",
    "artifact_quality": "Are objects free of distortions, melted details, and garbled text?",
    "overall_usability": "Could a designer use this image as material for a poster/mock-up/illustration?",
}
DIMENSIONS = list(HUMAN_RUBRIC)


def build_human_rubric() -> dict:
    return {"scale": ANCHORS, "dimensions": HUMAN_RUBRIC}


def export_annotation_sheets(records: list[dict], out_dir: str | Path, n_annotators: int = 2,
                             overlap_frac: float = 0.3, seed: int = 0) -> None:
    """Write blind per-annotator CSVs plus a private key file.

    Annotators see only an opaque item id, the prompt, and the image path in shuffled order;
    model names are hidden. An `overlap_frac` subset is shared by all annotators for agreement.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = list(range(len(records)))
    rng.shuffle(order)
    item_ids = {idx: f"item{n:04d}" for n, idx in enumerate(order)}
    n_overlap = int(len(records) * overlap_frac)
    overlap = set(order[:n_overlap])
    rest = order[n_overlap:]
    assigned = {a: set(overlap) for a in range(n_annotators)}
    for k, idx in enumerate(rest):
        assigned[k % n_annotators].add(idx)

    with open(out / "annotation_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "prompt_id", "model", "image_path", "is_overlap"])
        for idx in order:
            r = records[idx]
            w.writerow([item_ids[idx], r["prompt_id"], r["model"], r["image_path"], idx in overlap])
    for a in range(n_annotators):
        with open(out / f"annotator_{a + 1}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "prompt", "image_path", *DIMENSIONS])
            for idx in sorted(assigned[a], key=lambda i: item_ids[i]):
                w.writerow([item_ids[idx], records[idx]["prompt"], records[idx]["image_path"],
                            *[""] * len(DIMENSIONS)])


def load_ratings(path: str | Path) -> dict[str, dict[str, int]]:
    ratings: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(row[d].strip() for d in DIMENSIONS):
                ratings[row["item_id"]] = {d: int(row[d]) for d in DIMENSIONS}
    return ratings


def cohens_kappa(a: list[int], b: list[int], weights: str | None = "quadratic") -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, weights=weights, labels=[1, 2, 3, 4, 5]))


def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's alpha (interval metric). matrix: raters x items, NaN = missing."""
    values = [col[~np.isnan(col)] for col in matrix.T]
    values = [v for v in values if len(v) >= 2]
    if not values:
        return float("nan")
    n = sum(len(v) for v in values)
    d_o = sum(sum((x - y) ** 2 for x, y in combinations(v, 2)) * 2 / (len(v) - 1) for v in values) / n
    allv = np.concatenate(values)
    d_e = sum((x - y) ** 2 for x, y in combinations(allv, 2)) * 2 / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e else float("nan")


def agreement_report(sheet_paths: list[str | Path]) -> dict:
    """Weighted kappa (pairwise mean) and Krippendorff's alpha per rubric dimension on overlapping items."""
    sheets = [load_ratings(p) for p in sheet_paths]
    shared = sorted(set.intersection(*[set(s) for s in sheets]))
    report = {"n_shared_items": len(shared)}
    for d in DIMENSIONS:
        cols = [[s[i][d] for i in shared] for s in sheets]
        kappas = [cohens_kappa(x, y) for x, y in combinations(cols, 2)] if shared else []
        report[d] = {"weighted_kappa": float(np.mean(kappas)) if kappas else None,
                     "krippendorff_alpha": krippendorff_alpha_interval(np.array(cols, dtype=float))
                     if shared else None}
    return report
