# t2i_eval: offline benchmark for PromptCraft

This repo implements the **offline benchmark** in the paper *PromptCraft: An Evaluation-Driven Agentic Chatbot for Text-to-Image Creation* (`cvpr.tex`). In the paper's architecture it is the "Offline benchmark (5 categories, 2 models)" box that feeds the knowledge base, and it provides the Phase 2 evaluation protocol. The agent, retriever and chat front end are not part of this repo.

## Paper-to-code map

| Paper | Code |
| --- | --- |
| Five categories: counting, spatial relations, color/attribute binding, text rendering, compound scenes; 100-200 prompts; fixed seeds | `dataset.py` (5 x 30 = 150 prompts, one fixed seed per prompt shared across models) |
| JSON element checklist (`objects`, `relations`, `text`, `style`) | `dataset.py`: `checklist` per prompt, flattened to atomic `elements` |
| Two models: SDXL and SDXL-Turbo (latency/quality trade-off) | `generation.py` |
| Judge returns (element, satisfied, reason) per element | `judge.py` |
| In-loop judge differs from the evaluation judge | `judge.py`: `process_judge_batch(..., loop_judge_name)` raises if equal |
| CLIPScore and aesthetic score | `metrics.py` (CLIPScore as in Hessel et al.; aesthetic is a heuristic proxy, see below) |
| Model-capability profile for RAG (e.g. "accuracy drops for counts above 3") | `analysis.py`: `capability_profile` -> `outputs/capability_profile.json` |
| 1-5 rubric (instruction following, composition, artifact quality, usability); blind, randomized; overlap samples; Cohen's kappa / Krippendorff's alpha | `human_eval.py` |
| Spearman correlation between automatic metrics and humans | `analysis.py`: `metric_human_correlation` |
| Element pass rate by category, CLIPScore, aesthetic, latency | `analysis.py`: `category_table` |

Not covered here: conditions B-D (one-shot rewrite, agent with/without RAG), rounds and cost. These need the PromptCraft agent; condition A (direct prompt) is what this repo produces.

## Usage

```bash
pip install -e .                              # or: pip install -r requirements.txt && export PYTHONPATH=src
python -m t2i_eval bank                       # data/prompt_bank.json
python -m t2i_eval generate --models sdxl sdxl-turbo
python -m t2i_eval metrics
export OPENAI_API_KEY=...
python -m t2i_eval judge --judge-model <eval judge> --loop-judge-model <judge used in the agent loop>
python -m t2i_eval human-export               # blind CSVs in outputs/human/
python -m t2i_eval analyze --human-sheets outputs/human/annotator_1.csv outputs/human/annotator_2.csv
```

All stages read and write `outputs/records.json`. Generation needs a GPU for practical speed.

## Testing

```bash
python -m pytest tests                                  # unit tests (no GPU or API needed)
python -m t2i_eval judge --judge-model mock             # offline pipeline smoke test; mock scores are meaningless
```

Other robustness features: generation resumes from images already on disk; judge failures (bad JSON, API errors) are recorded as `judge_error` and excluded from aggregates instead of being scored; the category table reports bootstrap 95% CIs because each category has only ~30 prompts.

## Known limitations

- `compute_aesthetic_score` is a simple image-statistics heuristic, not a learned aesthetic model. Use the LAION aesthetic predictor before reporting final numbers.
- The judge client uses an OpenAI-compatible vision API; there is no offline fallback, so no placeholder scores are ever produced.
- No measured results are included yet; run the pipeline to produce them.
