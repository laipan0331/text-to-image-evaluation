# T2I Evaluation Framework

A compact text-to-image (T2I) evaluation framework inspired by GenEval, T2I-CompBench, and VBench.

## Overview

- Builds a benchmark with 100-200 prompts spanning key failure modes:
  - object count
  - spatial relations
  - color attributes
  - text rendering
  - composition/style
  - negation / attribute binding
- Generates images with open-source diffusion models using Hugging Face Diffusers.
- Computes automatic metrics such as CLIPScore and heuristic aesthetic scoring.
- Uses a multimodal LLM as a judge with a JSON-based rubric.
- Provides a human annotation workflow with agreement analysis and correlation study.
- Produces a modular, reproducible project structure for evaluation experiments.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── t2i_eval/
│       ├── __init__.py
│       ├── dataset.py
│       ├── generation.py
│       ├── metrics.py
│       ├── judge.py
│       ├── human_eval.py
│       └── run_eval.py
└── data/
    └── README.md
```

## Quick start

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.\.venv\Scripts\activate   # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate a prompt bank and evaluate:

```bash
python -m src.t2i_eval.run_eval --generate --evaluate --output-dir outputs
```

4. Inspect results:

```bash
ls outputs
```

## Prompt benchmark design

The default benchmark contains 120 prompts distributed across categories such as:

- object_count
- spatial_relation
- color_attribute
- text_rendering
- style_composition
- attribute_binding

Each category includes multiple templates with acceptable variations to reduce overfitting.

## Automatic metrics

- CLIPScore: image-text alignment
- heuristic aesthetic scoring: image quality / pleasantness estimate based on image statistics
- MLLM judge: rubric-based instruction following, structured as JSON

## Human evaluation workflow

- define rubric items (instruction following, composition, detail quality, overall aesthetics)
- collect 2-3 annotators on a subset of generated images
- compute agreement via Cohen's kappa or Krippendorff's alpha
- compare human scores against automatic metrics

## Results

Results are written to `outputs/` after running the generation and evaluation
pipeline. The table below is an illustrative expected baseline for the first
benchmark run, not a measured result. It will be replaced with per-category
scores after the generation pipeline completes successfully.

| Category | CLIP alignment | Heuristic aesthetic score | Judge overall quality |
| --- | ---: | ---: | ---: |
| object_count | 0.62 | 0.58 | 0.64 |
| spatial_relation | 0.59 | 0.61 | 0.60 |
| color_attribute | 0.70 | 0.66 | 0.73 |
| text_rendering | 0.44 | 0.57 | 0.46 |
| style_composition | 0.76 | 0.72 | 0.78 |
| attribute_binding | 0.63 | 0.64 | 0.65 |

The expected pattern is stronger performance on style and color prompts, with
lower scores on text rendering and precise spatial relations. The heuristic
aesthetic score is intentionally treated as a lightweight proxy rather than a
pretrained aesthetic model.

## Notes

Designed to be lightweight and reproducible, prioritizing clarity of evaluation methodology.

## Example benchmark references

- GenEval
- T2I-CompBench
- VBench
