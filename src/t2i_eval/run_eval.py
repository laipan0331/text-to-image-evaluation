"""Staged CLI. Each stage reads/writes outputs/records.json.

    python -m t2i_eval bank
    python -m t2i_eval generate --models sdxl sdxl-turbo
    python -m t2i_eval metrics
    python -m t2i_eval judge --judge-model gpt-4o --loop-judge-model <in-loop judge>
    python -m t2i_eval human-export
    python -m t2i_eval analyze [--human-sheets outputs/human/annotator_1.csv ...]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dataset import build_prompt_bank, load_json, save_json


def main(argv=None):
    p = argparse.ArgumentParser(description="T2I benchmark for PromptCraft")
    p.add_argument("stage", choices=["bank", "generate", "metrics", "judge", "human-export", "analyze"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--prompt-bank", default="data/prompt_bank.json")
    p.add_argument("--models", nargs="+", default=["sdxl", "sdxl-turbo"])
    p.add_argument("--num-per-category", type=int, default=30)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--loop-judge-model", default=None, help="judge used inside the agent loop; must differ")
    p.add_argument("--human-sheets", nargs="*", default=[])
    p.add_argument("--n-annotators", type=int, default=2)
    a = p.parse_args(argv)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.json"

    if a.stage == "bank":
        bank = build_prompt_bank(a.num_per_category, a.prompt_bank)
        print(f"Wrote {len(bank)} prompts to {a.prompt_bank}")
        return

    if a.stage == "generate":
        from .generation import generate_images

        bank = load_json(a.prompt_bank)
        rows = []
        for m in a.models:
            rows += generate_images(bank, m, out / "images")
        save_json(records_path, rows)
        print(f"Generated {len(rows)} images.")
        return

    records = load_json(records_path)

    if a.stage == "metrics":
        from .metrics import compute_metrics_for_records

        save_json(records_path, compute_metrics_for_records(records))
    elif a.stage == "judge":
        from .judge import MockJudge, OpenAIVisionJudge, process_judge_batch

        if not a.judge_model:
            p.error("--judge-model is required for the judge stage (use 'mock' for a smoke test only)")
        client = MockJudge() if a.judge_model == "mock" else OpenAIVisionJudge(a.judge_model)
        save_json(records_path, process_judge_batch(records, client, a.loop_judge_model))
    elif a.stage == "human-export":
        from .human_eval import export_annotation_sheets

        export_annotation_sheets(records, out / "human", a.n_annotators)
        print(f"Sheets written to {out / 'human'}")
    elif a.stage == "analyze":
        from .analysis import capability_profile, category_table, metric_human_correlation
        from .human_eval import agreement_report, load_ratings

        result = {"category_table": category_table(records)}
        (out / "capability_profile.json").write_text(
            json.dumps(capability_profile(records), indent=2), encoding="utf-8")
        if a.human_sheets:
            sheets = [load_ratings(s) for s in a.human_sheets]
            ids = set().union(*sheets)
            mean = {i: {d: sum(s[i][d] for s in sheets if i in s) / sum(i in s for s in sheets)
                        for d in next(s[i] for s in sheets if i in s)} for i in ids}
            with open(out / "human" / "annotation_key.csv", newline="", encoding="utf-8") as f:
                key = {r["item_id"]: r for r in csv.DictReader(f)}
            result["agreement"] = agreement_report(a.human_sheets)
            result["metric_human_spearman"] = metric_human_correlation(records, mean, key)
        (out / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote analysis.json and capability_profile.json to {out}")


if __name__ == "__main__":
    main()
