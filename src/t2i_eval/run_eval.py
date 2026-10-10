from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_prompt_bank, load_prompt_bank, save_prompt_bank
from .generation import generate_images
from .metrics import compute_metrics_for_records, save_metric_report
from .judge import process_judge_batch


def main():
    parser = argparse.ArgumentParser(description="Text-to-image evaluation pipeline")
    parser.add_argument("--generate", action="store_true", help="generate prompts and images")
    parser.add_argument("--evaluate", action="store_true", help="compute automatic metrics")
    parser.add_argument("--prompt-bank", type=str, default="data/prompt_bank.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num-per-category", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        prompt_bank = build_prompt_bank(num_per_category=args.num_per_category, output_path=args.prompt_bank)
        generated = generate_images(prompt_bank, model_name=args.model, output_dir=output_dir / "images", num_per_prompt=2)
        save_prompt_bank(args.prompt_bank, generated)
        print(f"Generated {len(generated)} images.")

    if args.evaluate:
        prompt_bank_path = Path(args.prompt_bank)
        if prompt_bank_path.exists():
            records = load_prompt_bank(prompt_bank_path)
        else:
            records = build_prompt_bank(num_per_category=args.num_per_category)
        metrics = compute_metrics_for_records(records)
        metric_path = output_dir / "metrics.json"
        save_metric_report(metrics, metric_path)

        judge_results = process_judge_batch(metrics)
        judge_path = output_dir / "judge_results.json"
        judge_path.write_text(json.dumps(judge_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved metric and judge results to {output_dir}.")


if __name__ == "__main__":
    main()
