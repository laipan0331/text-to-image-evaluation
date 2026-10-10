# T2I Evaluation Framework

A compact text-to-image (T2I) evaluation framework inspired by GenEval, T2I-CompBench, and VBench.

## Goals

- Build a small benchmark with 100-200 prompts spanning key failure modes:
  - object count
  - spatial relations
  - color attributes
  - text rendering
  - composition/style
  - negation / attribute binding
- Generate images with two open-source diffusion models using Hugging Face Diffusers.
- Compute automatic metrics such as CLIPScore and aesthetic score.
- Use a multimodal LLM as a judge with a JSON-based rubric.
- Add a human annotation workflow with agreement analysis and correlation study.
- Produce a clean project structure suitable for interviews and experiments.

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
- aesthetic score: image quality / pleasantness
- MLLM judge: rubric-based instruction following, structured as JSON

## Human evaluation workflow

- define rubric items (instruction following, composition, detail quality, overall aesthetics)
- collect 2-3 annotators on a subset of generated images
- compute agreement via Cohen's kappa or Krippendorff's alpha
- compare human scores against automatic metrics

## Results

Results are written to `outputs/` after running the generation and evaluation
pipeline. The benchmark report should include per-category scores for CLIP
alignment, the image-based aesthetic heuristic, and the judge rubric, together
with representative generated images and a short analysis of metric agreement.

| Category | CLIP alignment | Aesthetic score | Judge overall quality |
| --- | ---: | ---: | ---: |
| object_count | pending | pending | pending |
| spatial_relation | pending | pending | pending |
| color_attribute | pending | pending | pending |
| text_rendering | pending | pending | pending |
| style_composition | pending | pending | pending |
| attribute_binding | pending | pending | pending |

The first reproducible run is still pending because generated image artifacts
have not yet been produced in this checkout.

## Notes

Designed to be lightweight and reproducible, prioritizing clarity of evaluation methodology.

## Example benchmark references

- GenEval
- T2I-CompBench
- VBench
