"""Element-level multimodal judge (paper Sec. 4, contributions 1 and 4).

The judge returns one (element, satisfied, reason) record per checklist element.
The judge used for the final evaluation must differ from the one used inside the
PromptCraft loop; `process_judge_batch` enforces this.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path


def build_judge_prompt(prompt: str, elements: list[dict]) -> str:
    items = "\n".join(f"{e['id']}. {e['text']}" for e in elements)
    return (
        "You are a strict evaluator of a text-to-image result. Look at the image and check each element.\n"
        f"Original prompt: {prompt}\n\nElements:\n{items}\n\n"
        'Reply with JSON only: {"verdicts": [{"id": <int>, "satisfied": true|false, "reason": "<short>"}], '
        '"overall_quality": <1-5 integer, visual quality only>}'
    )


class OpenAIVisionJudge:
    """Judge backed by any OpenAI-compatible vision chat endpoint (reads OPENAI_API_KEY)."""

    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def generate(self, prompt: str, image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        resp = self.client.chat.completions.create(
            model=self.name, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}])
        return resp.choices[0].message.content


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def judge_record(record: dict, client) -> dict:
    raw = client.generate(build_judge_prompt(record["prompt"], record["elements"]), record["image_path"])
    parsed = _parse_json(raw)
    by_id = {int(v["id"]): v for v in parsed.get("verdicts", [])}
    verdicts = []
    for el in record["elements"]:
        v = by_id.get(el["id"])
        verdicts.append({"id": el["id"], "kind": el["kind"], "element": el["text"],
                         "count": el.get("count"),
                         "satisfied": bool(v["satisfied"]) if v else False,
                         "reason": v.get("reason", "") if v else "missing from judge output"})
    n = max(len(verdicts), 1)
    return {"verdicts": verdicts,
            "element_pass_rate": sum(v["satisfied"] for v in verdicts) / n,
            "all_pass": all(v["satisfied"] for v in verdicts),
            "judge_overall_quality": parsed.get("overall_quality"),
            "judge_name": getattr(client, "name", "unknown")}


def process_judge_batch(records: list[dict], client, loop_judge_name: str | None = None) -> list[dict]:
    if loop_judge_name and getattr(client, "name", None) == loop_judge_name:
        raise ValueError("Evaluation judge must differ from the in-loop judge "
                         f"(both are '{loop_judge_name}').")
    return [{**r, **judge_record(r, client)} for r in records]
