import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_eval.analysis import bootstrap_ci, capability_profile, category_table  # noqa: E402
from t2i_eval.dataset import CATEGORIES, build_prompt_bank  # noqa: E402
from t2i_eval.human_eval import cohens_kappa, export_annotation_sheets, krippendorff_alpha_interval  # noqa: E402
from t2i_eval.judge import MockJudge, process_judge_batch  # noqa: E402


@pytest.fixture(scope="module")
def bank():
    return build_prompt_bank(30)


def test_bank_shape_and_determinism(bank):
    assert len(bank) == 150
    assert {b["category"] for b in bank} == set(CATEGORIES)
    assert len({b["prompt"] for b in bank if b["category"] == "counting"}) == 30
    assert bank == build_prompt_bank(30)
    assert all(b["elements"] for b in bank)


def test_checklist_elements(bank):
    text = next(b for b in bank if b["category"] == "text_rendering")
    assert [e["kind"] for e in text["elements"]] == ["text"]
    counting = next(b for b in bank if b["category"] == "counting")
    assert counting["elements"][0]["kind"] == "count"


def test_judge_roundtrip_and_error_handling(bank):
    recs = [{**b, "model": "sdxl", "image_path": f"img{b['prompt_id']}.png"} for b in bank[:10]]
    out = process_judge_batch(recs, MockJudge())
    assert all(0 <= r["element_pass_rate"] <= 1 for r in out)

    class Broken:
        name = "broken"

        def generate(self, p, i):
            return "not json"

    bad = process_judge_batch(recs[:1], Broken())[0]
    assert bad["element_pass_rate"] is None and "judge_error" in bad
    assert category_table(bad and [bad])  # errored records do not crash aggregation


def test_judge_must_differ_from_loop_judge(bank):
    with pytest.raises(ValueError):
        process_judge_batch([], MockJudge(), loop_judge_name="mock")


def test_capability_profile_counts():
    rec = {"model": "m", "verdicts": [
        {"kind": "count", "count": 2, "satisfied": True},
        {"kind": "count", "count": 5, "satisfied": False}]}
    notes = capability_profile([rec])["m"]["notes"][0]
    assert "1.00" in notes and "0.00" in notes


def test_agreement_stats():
    assert cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert krippendorff_alpha_interval(np.array([[1, 2, 3, 4.], [1, 2, 3, 4.]])) == pytest.approx(1.0)
    lo, hi = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 1.0])
    assert 0 <= lo <= hi <= 1


def test_blind_export(bank, tmp_path):
    recs = [{**b, "model": "sdxl", "image_path": "x.png"} for b in bank[:20]]
    export_annotation_sheets(recs, tmp_path, n_annotators=2, overlap_frac=0.3)
    sheet = (tmp_path / "annotator_1.csv").read_text(encoding="utf-8")
    assert "sdxl" not in sheet
    key = (tmp_path / "annotation_key.csv").read_text(encoding="utf-8")
    assert key.count("True") == 6
