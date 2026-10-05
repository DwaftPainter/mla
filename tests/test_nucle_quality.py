import json

import pandas as pd

from src.data.constants import TARGET_LABELS
from src.data.data_quality import annotation_review, audit_quality, compare_csv, duplicate_content
from src.data.nucle_sgml import NucleDocument, NucleMistake
from src.data.parse_nucle import documents_to_dataframe


def mistake(start, end, label="Nn", correction="fixed"):
    return NucleMistake(0, start, 0, end, label, correction, missing_correction=False)


def test_audit_retains_overlaps_duplicate_candidates_and_different_labels():
    annotations = [mistake(0, 7), mistake(5, 10, "SVA"),
                   mistake(5, 10, "Prep"), mistake(0, 7)]
    frame = documents_to_dataframe([NucleDocument("x", ["Cats sleep."], annotations)])
    before = frame.copy(deep=True)
    audit, flags = annotation_review(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert audit.annotation_id.tolist() == [0, 1, 2, 3]
    assert audit.error_type.tolist() == ["Nn", "SVA", "Prep", "Nn"]
    assert audit.same_span_different_label.tolist() == [False, True, True, False]
    assert audit.duplicate_candidate.tolist() == [True, False, False, True]
    assert audit.overlapping_span.all()
    assert json.loads(audit.iloc[0].overlap_annotation_ids) == [1, 2, 3]
    assert audit.review_status.tolist() == ["pending"] * 4


def test_insertion_review_and_touching_spans():
    frame = documents_to_dataframe([NucleDocument("x", ["abcdef"], [
        mistake(0, 3), mistake(3, 6), mistake(1, 1, "ArtOrDet"),
        mistake(1, 1, "Prep"), mistake(3, 3),
    ])])
    audit, _ = annotation_review(frame)
    assert [json.loads(value) for value in audit.overlap_annotation_ids] == [[2, 3], [], [0, 3], [0, 2], []]
    assert audit.same_span_different_label.tolist() == [False, False, True, True, False]


def test_duplicate_essay_ids_do_not_create_false_annotation_overlap():
    docs = [NucleDocument("same", ["Cats."], [mistake(0, 4)]),
            NucleDocument("same", ["Cats."], [mistake(0, 4, "SVA")])]
    audit, _ = annotation_review(documents_to_dataframe(docs))
    assert not audit.overlapping_span.any()
    assert not audit.same_span_different_label.any()
    assert not audit.duplicate_candidate.any()


def test_invalid_offset_and_missing_correction_are_flags_not_removal():
    frame = documents_to_dataframe([NucleDocument("x", ["Cats."], [
        mistake(0, 99), NucleMistake(0, 0, 0, 4, "Nn"), mistake(0, 4, correction=""),
    ])])
    audit, _ = annotation_review(frame)
    assert len(audit) == len(frame) == 3
    assert audit.invalid_offset.tolist() == [True, False, False]
    assert audit.missing_correction.tolist() == [False, True, False]
    assert frame.edit_type.tolist() == ["invalid_offset", "missing_correction", "deletion"]


def test_csv_roundtrip_and_corruption(tmp_path):
    frame = documents_to_dataframe([NucleDocument("001", ["Cats."], [mistake(0, 4)])])
    path = tmp_path / "annotations.csv"
    assert compare_csv(frame, path)["status"] == "missing"
    frame.to_csv(path, index=False)
    assert compare_csv(frame, path)["status"] == "match"
    changed = frame.copy()
    changed.loc[0, "error_type"] = "Prep"
    changed.to_csv(path, index=False)
    assert compare_csv(frame, path)["different_cells"] == 1
    assert compare_csv(frame, path)["status"] == "mismatch"


def test_content_duplicate_audit_does_not_create_fold_or_group_manifest(tmp_path):
    docs = [NucleDocument("a", ["References", "Some CONTENT."]),
            NucleDocument("b", ["References", "some content."])]
    records = duplicate_content(docs)
    paragraphs = [item for item in records if item["scope"] == "paragraph"]
    assert sum(item["kind"] == "exact" for item in paragraphs) == 1
    assert sum(item["kind"] == "normalized" for item in paragraphs) == 2
    summary, audit, _ = audit_quality(docs, documents_to_dataframe(docs), tmp_path / "missing.csv")
    assert summary["documents"] == 2
    assert summary["annotations"] == 0
    assert summary["missing_target_classes"] == list(TARGET_LABELS)
    assert not {"group_id", "fold", "pos", "sentence"} & set(audit.columns)


def test_empty_quality_schema(tmp_path):
    summary, audit, duplicates = audit_quality([], documents_to_dataframe([]), tmp_path / "missing.csv")
    assert summary["annotations"] == 0
    assert summary["overlap_pairs"] == 0
    assert audit.empty and duplicates == []
    assert "review_reasons" in audit
