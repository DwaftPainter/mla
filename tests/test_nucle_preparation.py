"""Boundary, leakage, source identity and publication checks for preparation v1."""
import json

import pytest

from src.data.nucle_sgml import NucleDocument, NucleMistake, load_nucle
from src.data.parse_nucle import documents_to_dataframe
from src.data.data_quality import annotation_review
from src.data.prepare_nucle import (
    build_dataset, canonical, construct_input, digest, export_dataset, file_hash,
    make_nlp, prepare, read_metadata, segment_paragraph,
)


@pytest.fixture(scope="module")
def nlp():
    return make_nlp()


@pytest.mark.parametrize("text,expected", [
    ("Dr. Smith arrived. He left.", ["Dr. Smith arrived. ", "He left."]),
    ("The value is 3.14. Fine!", ["The value is 3.14. ", "Fine!"]),
    ("I can't go. She won't.", ["I can't go. ", "She won't."]),
    ('"Go!" (He left.) Next.', ['"Go!" (', 'He left.) ', 'Next.']),
    ("Wait... Fine.", ["Wait... Fine."]),
    ("Wait… Fine.", ["Wait… ", "Fine."]),
    ("a\nb; c: d", ["a\nb; c: d"]),
    ("  Hi.  She go.\t", ["  Hi.  ", "She go.\t"]),
    ("Title without punctuation", ["Title without punctuation"]),
    (" \t\n", []), ("", []),
])
def test_locked_sentence_boundaries(nlp, text, expected):
    assert [s["sentence"] for s in segment_paragraph(text, nlp, policy="nucle-preparation-v1")] == expected


def test_unicode_literal_entities_and_parser_roundtrip(nlp, tmp_path):
    path = tmp_path / "fixture.sgml"
    path.write_bytes('<DOC nid="x"><TEXT><TITLE>Title</TITLE><P>\r\n  Cafe\u0301 😀 &amp; A&A.\tNext  line\r\nend.\r\n</P></TEXT></DOC>'.encode())
    doc = load_nucle(path)[0]
    assert doc.paragraphs == ["Title", "  Cafe\u0301 😀 &amp; A&A.\tNext  line\nend."]
    paragraph = doc.paragraphs[1]
    sentences = segment_paragraph(paragraph, nlp)
    assert "".join(s["sentence"] for s in sentences) == paragraph
    for sentence in sentences:
        for token in sentence["tokens"]:
            assert sentence["sentence"][token["start"]:token["end"]] == token["text"]
    start = paragraph.index("😀")
    item, _, reason = construct_input(paragraph, start, start + 1, sentences)
    assert reason is None and item["error_span"] == "😀"


@pytest.mark.parametrize("start,end,span,alignment,reason", [
    (3, 4, "s", "partial_token", None),
    (0, 5, "Dogs ", "whitespace_edge", None),
    (0, 4, "Dogs", "exact", None),
    (0, 8, "Dogs run", "exact", None),
    (8, 9, ".", "exact", None),
    (4, 5, None, None, "no_token_overlap"),
    (2, 2, None, None, "insertion_inside_token"),
    (4, 4, "", "insertion_boundary", None),
    (0, 0, "", "insertion_boundary", None),
    (9, 9, "", "insertion_boundary", None),
])
def test_alignment(nlp, start, end, span, alignment, reason):
    text = "Dogs run."
    item, sent, actual_reason = construct_input(text, start, end, segment_paragraph(text, nlp))
    assert actual_reason == reason
    assert sent is not None
    if item is not None:
        assert item["error_span"] == span and item["alignment"] == alignment
        if alignment == "partial_token":
            assert item["span_token_indices"] == [0]
            assert item["left_token_indices"] == []
            assert item["right_token_indices"] == [1, 2]
            assert item["cuts_start_token"] and not item["cuts_end_token"]


def test_sentence_conversion_and_insertion_ties(nlp):
    text = "Hi. She go."
    sentences = segment_paragraph(text, nlp)
    item, _, reason = construct_input(text, 8, 10, sentences)
    assert reason is None
    assert (item["sentence"], item["error_start"], item["error_end"], item["error_span"]) == ("She go.", 4, 6, "go")
    assert construct_input(text, 2, 5, sentences)[2] == "cross_sentence"
    for position, index, offset in [(4, 1, 0), (11, 1, 7), (3, 0, 3)]:
        item, sent, reason = construct_input(text, position, position, sentences)
        assert reason is None and sent["sentence_index"] == index and item["error_start"] == offset
    assert construct_input(text, 0, 4, sentences)[0]["error_span"] == "Hi. "


def dataset_for(nlp, docs):
    frame = documents_to_dataframe(docs)
    flags = {int(aid): [] for aid in frame.annotation_id}
    return build_dataset(docs, frame, flags, {0: "unresolved"}, "fixture", nlp)


def test_reason_priority_and_full_accounting(nlp):
    doc = NucleDocument("x", ["Dogs run.", " \t", "Hi. She go."], [
        NucleMistake(9, 99, 9, 99, "Other"),
        NucleMistake(0, 99, 1, 0, "Nn"),
        NucleMistake(0, 0, 1, 0, "Nn"),
        NucleMistake(1, 0, 1, 0, "Nn"),
        NucleMistake(2, 2, 2, 5, "Nn"),
        NucleMistake(0, 4, 0, 5, "Nn"),
        NucleMistake(0, 2, 0, 2, "Nn"),
        NucleMistake(0, 0, 0, 4, "Nn"),
    ])
    data = dataset_for(nlp, [doc])
    assert [r["reason"] for r in data["provenance"]] == [
        "label_out_of_scope", "invalid_offset", "cross_paragraph", "empty_paragraph",
        "cross_sentence", "no_token_overlap", "insertion_inside_token", None]
    assert data["summary"]["counts"] == {"all": 8, "target": 7, "out_of_scope": 1, "retained": 1, "not_constructible": 6}
    assert data["samples"][0]["annotation_id"] == 7


def test_overlap_unresolved_ambiguity_and_no_correction_leakage(nlp):
    docs = [NucleDocument("x", ["Dogs run."], [
        NucleMistake(0, 0, 0, 4, "Nn", "Dog", "comment"),
        NucleMistake(0, 0, 0, 4, "SVA", "changed"),
        NucleMistake(0, 3, 0, 4, "Nn"),
    ])]
    original = dataset_for(nlp, docs)
    assert len(original["samples"]) == 3
    assert original["provenance"][0]["review_status"] == "unresolved"
    assert original["summary"]["input_duplicates"]["ambiguous_groups"] == 1
    docs[0].mistakes[0].correction = "entirely different answer"
    docs[0].mistakes[0].comment = "secret"
    docs[0].mistakes[0].error_type = "Prep"
    changed = dataset_for(nlp, docs)
    assert changed["sentences"] == original["sentences"]
    assert changed["samples"][0]["input_sha256"] == original["samples"][0]["input_sha256"]
    assert set(changed["samples"][0]).isdisjoint({"correction", "comment", "review_status", "quality_flags"})


def make_source_artifacts(tmp_path):
    source = tmp_path / "source.sgml"
    source.write_text('<DOC nid="x"><TEXT><P>Dogs run.</P></TEXT><ANNOTATION>'
                      '<MISTAKE start_par="0" start_off="0" end_par="0" end_off="4">'
                      '<TYPE>Nn</TYPE><CORRECTION>Dog</CORRECTION></MISTAKE>'
                      '</ANNOTATION></DOC>', encoding="utf-8")
    frame = documents_to_dataframe(load_nucle(source))
    annotations, audit, review = [tmp_path / f"{name}.csv" for name in ("annotations", "audit", "review")]
    frame.to_csv(annotations, index=False)
    annotation_review(frame)[0].to_csv(audit, index=False)
    review.write_text("annotation_id,essay_id,error_type,source_sha256,decision,data_action\n"
                      f"0,x,Nn,{file_hash(source)},unresolved,keep_original_annotation_unchanged\n", encoding="utf-8")
    return source, annotations, audit, review, frame


def test_metadata_rejects_stale_audit_and_foreign_review(tmp_path):
    source, _, audit, review, frame = make_source_artifacts(tmp_path)
    assert read_metadata(frame, audit, review, file_hash(source))[1] == {0: "unresolved"}
    review.write_text(review.read_text().replace(file_hash(source), "wrong"))
    with pytest.raises(ValueError, match="checksum"):
        read_metadata(frame, audit, review, file_hash(source))
    audit.write_text(audit.read_text().replace("Nn", "Prep"))
    with pytest.raises(ValueError, match="Audit CSV"):
        read_metadata(frame, audit, review, file_hash(source))


def test_complete_run_reproducibility_and_no_overwrite(tmp_path):
    source, annotations, audit, review, _ = make_source_artifacts(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        prepare(source, annotations, audit, review, output, file_hash(source))
    a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert a["outputs"] == b["outputs"]
    assert (first / "manifest.sha256").read_text().strip() == digest(canonical(a).encode())
    for name, expected in a["outputs"].items():
        assert file_hash(first / name) == expected
    before = file_hash(first / "samples.jsonl")
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, annotations, audit, review, first, file_hash(source))
    assert file_hash(first / "samples.jsonl") == before
    with pytest.raises(ValueError, match="Source checksum"):
        prepare(source, annotations, audit, review, tmp_path / "bad", "wrong")
    assert not (tmp_path / "bad").exists()
    annotations.write_text(annotations.read_text().replace("Dogs run.", "Cats run."))
    with pytest.raises(ValueError, match="Understanding CSV"):
        prepare(source, annotations, audit, review, tmp_path / "bad", file_hash(source))


def test_changed_input_prevents_publication(tmp_path):
    source = tmp_path / "input"
    source.write_text("changed")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Input changed"):
        export_dataset(output, {"samples": [], "summary": {}}, {}, {source: "old-hash"})
    assert not output.exists()
