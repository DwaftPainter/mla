"""Regression checks for source paragraphs whose sentence boundary is ambiguous."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.boundary_challenges import BOUNDARY_CHALLENGES
from src.data.nucle_sgml import NucleDocument
from src.data.prepare_nucle import build_dataset, make_nlp, segment_paragraph

FIXTURE = Path(__file__).parent / "fixtures" / "nucle_boundary_challenges.jsonl"
if not FIXTURE.exists():
    pytest.skip("Licensed NUCLE boundary fixture is local only; see tests/README.md", allow_module_level=True)
CHALLENGES = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def nlp():
    return make_nlp()


@pytest.mark.parametrize("challenge", CHALLENGES, ids=lambda row: str(row["annotation_id"]))
def test_ambiguous_boundary_challenge_keeps_v2_output_and_flag(nlp, challenge):
    aid = challenge["annotation_id"]
    policy = BOUNDARY_CHALLENGES[aid]
    assert challenge["boundary_flags"] == [policy["flag"]]
    assert challenge["boundary_review_status"] == policy["review_status"] == "unresolved"
    assert challenge["expectation"] == policy["expectation"]

    actual = segment_paragraph(challenge["paragraph"], nlp)
    observed = [{"start": row["sentence_start_par"],
                 "end": row["sentence_end_par"],
                 "text": row["sentence"]} for row in actual]
    assert observed == challenge["expected_paragraph_sentences"]
    focus = next(row for row in actual
                 if row["sentence_start_par"] == challenge["expected_sentence_start_par"]
                 and row["sentence_end_par"] == challenge["expected_sentence_end_par"])
    assert focus["sentence"] == challenge["expected_sentence"]
    assert challenge["paragraph"][challenge["annotation_start_par"]:challenge["annotation_end_par"]] == challenge["annotation_span"]


def test_build_dataset_keeps_and_flags_both_challenge_samples(nlp):
    rows = []
    for paragraph_index, challenge in enumerate(CHALLENGES):
        paragraph = challenge["paragraph"]
        start = challenge["annotation_start_par"]
        end = challenge["annotation_end_par"]
        rows.append({
            "annotation_id": challenge["annotation_id"],
            "essay_id": "challenge-fixture",
            "essay_index": 0,
            "mistake_index": challenge["annotation_id"],
            "start_par": paragraph_index,
            "end_par": paragraph_index,
            "start_off": start,
            "end_off": end,
            "error_type": "Nn",
            "valid_offset": True,
            "cross_paragraph": False,
            "paragraph": paragraph,
            "error_span": challenge["annotation_span"],
            "is_target_label": True,
        })
    paragraphs = [challenge["paragraph"] for challenge in CHALLENGES]
    frame = pd.DataFrame(rows)
    ids = {row["annotation_id"] for row in rows}
    built = build_dataset([NucleDocument("challenge-fixture", paragraphs)], frame,
                          {aid: [] for aid in ids}, {}, "fixture-source", nlp)

    assert {row["annotation_id"] for row in built["samples"]} == ids
    assert {row["annotation_id"] for row in built["boundary_challenges"]} == ids
    for table in (built["samples"], built["provenance"]):
        flagged = {row["annotation_id"]: row for row in table if row["annotation_id"] in ids}
        for challenge in CHALLENGES:
            row = flagged[challenge["annotation_id"]]
            assert row["boundary_flags"] == challenge["boundary_flags"]
            assert row["boundary_review_status"] == "unresolved"
