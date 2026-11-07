"""Aggregate tables, the model-capability profile for the RAG knowledge base, and
metric-vs-human correlation (paper Sec. 4, contributions 2 and 4)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .human_eval import DIMENSIONS


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def bootstrap_ci(xs, n_boot: int = 2000, seed: int = 0):
    """95% percentile bootstrap CI of the mean (categories only have ~30 prompts)."""
    xs = np.array([x for x in xs if x is not None], dtype=float)
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, (n_boot, len(xs))).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def category_table(records: list[dict]) -> dict:
    """Per model x category: element pass rate (+95% CI), all-pass rate, CLIPScore, aesthetic, latency."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["category"])].append(r)
    return {f"{m}/{c}": {
        "n": len(rs),
        "n_judge_errors": sum(1 for r in rs if r.get("judge_error")),
        "element_pass_rate": _mean(r.get("element_pass_rate") for r in rs),
        "element_pass_rate_ci95": bootstrap_ci(r.get("element_pass_rate") for r in rs),
        "prompt_all_pass_rate": _mean(None if r.get("all_pass") is None else float(r["all_pass"]) for r in rs),
        "clip_score": _mean(r.get("clip_score") for r in rs),
        "aesthetic_score": _mean(r.get("aesthetic_score") for r in rs),
        "latency_s": _mean(r.get("latency_s") for r in rs),
    } for (m, c), rs in sorted(groups.items())}


def capability_profile(records: list[dict], count_threshold: int = 3) -> dict:
    """Per-model profile that is indexed into the chatbot's knowledge base.

    Contains pass rate by element kind, pass rate by requested count, and
    human-readable notes such as 'accuracy drops for counts above 3'.
    """
    profile: dict = {}
    for model in sorted({r["model"] for r in records}):
        kind_hits = defaultdict(list)
        count_hits = defaultdict(list)
        for r in (x for x in records if x["model"] == model):
            for v in r.get("verdicts", []):
                kind_hits[v["kind"]].append(v["satisfied"])
                if v["kind"] == "count" and v.get("count"):
                    count_hits[v["count"]].append(v["satisfied"])
        by_kind = {k: {"pass_rate": _mean(map(float, h)), "n": len(h)} for k, h in kind_hits.items()}
        by_count = {str(k): {"pass_rate": _mean(map(float, h)), "n": len(h)}
                    for k, h in sorted(count_hits.items())}
        low = [x for k, h in count_hits.items() if k <= count_threshold for x in h]
        high = [x for k, h in count_hits.items() if k > count_threshold for x in h]
        notes = []
        if low and high:
            notes.append(f"counting: pass rate {np.mean(low):.2f} for counts <= {count_threshold} vs "
                         f"{np.mean(high):.2f} for counts > {count_threshold} (n={len(low)}/{len(high)}).")
        for k, s in sorted(by_kind.items(), key=lambda kv: kv[1]["pass_rate"]):
            notes.append(f"{k}: element pass rate {s['pass_rate']:.2f} (n={s['n']}).")
        profile[model] = {"by_element_kind": by_kind, "by_count": by_count, "notes": notes}
    return profile


def metric_human_correlation(records: list[dict], human: dict[str, dict[str, int]],
                             key: dict[str, dict]) -> dict:
    """Spearman correlation of automatic metrics against mean human ratings.

    `human`: item_id -> ratings averaged over annotators; `key`: item_id -> {prompt_id, model}.
    """
    from scipy.stats import spearmanr

    idx = {(r["prompt_id"], r["model"]): r for r in records}
    rows = []
    for item_id, ratings in human.items():
        k = key.get(item_id)
        r = idx.get((int(k["prompt_id"]), k["model"])) if k else None
        if r:
            rows.append((r, ratings))
    out: dict = {"n": len(rows)}
    metrics = ["clip_score", "aesthetic_score", "element_pass_rate", "judge_overall_quality"]
    for m in metrics:
        for d in DIMENSIONS:
            pairs = [(r[m], h[d]) for r, h in rows if r.get(m) is not None]
            if len(pairs) >= 3:
                rho, p = spearmanr(*zip(*pairs))
                out[f"{m}~{d}"] = {"spearman": float(rho), "p": float(p), "n": len(pairs)}
    return out
