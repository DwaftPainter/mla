import pytest
import spacy

from src.features.nucle_features import TrainOnlyEncoder, extract_features


def test_extractor_ignores_label_ids_correction_and_provenance():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    sentence = "She go to school."
    doc = nlp(sentence)
    start, end = sentence.index("go"), sentence.index("go") + 2
    base = {"sentence": sentence, "error_start": start, "error_end": end,
            "error_span": "go", "span_kind": "non_empty", "error_type": "SVA",
            "sample_id": "private-id", "essay_id": "private-essay", "correction": "goes",
            "comment": "review only", "alignment": "audit"}
    changed = {**base, "error_type": "Vform", "sample_id": "other-id", "essay_id": "other",
               "correction": "went", "comment": "different", "alignment": "different"}
    assert extract_features(base, doc) == extract_features(changed, doc)
    assert not any(any(term in key.lower() for term in ("label", "sample_id", "essay_id", "correction", "comment"))
                   for key in extract_features(base, doc))


def test_encoder_rejects_non_train_and_unknown_value_is_train_scoped():
    encoder = TrainOnlyEncoder().fit([
        {"sample_id": "t1", "partition": "train", "features": {"S|x": 1.0, "P|tag": "NOUN", "L|span_ngrams": "U cat"}},
    ])
    assert all("RARE_VALIDATION" not in name for name in encoder.feature_names)
    x = encoder.transform_one({"S|x": 2.0, "P|tag": "RARE_VALIDATION", "L|span_ngrams": "U dog"})
    assert x[encoder.index["C|P|tag|__UNK__"]] == 1.0
    assert x[encoder.index["L|span_ngrams|__UNK__"]] == 1.0
    with pytest.raises(ValueError, match="train rows only"):
        TrainOnlyEncoder().fit([{"sample_id": "v1", "partition": "validation", "features": {"S|x": 1.0}}])
    with pytest.raises(ValueError, match="Unexpected feature family"):
        TrainOnlyEncoder().fit([{"sample_id": "t2", "partition": "train", "features": {"error_type": "Prep"}}])


def test_offset_mismatch_fails_closed():
    doc = spacy.blank("en")("She goes.")
    with pytest.raises(ValueError, match="Span/source text mismatch"):
        extract_features({"sentence": doc.text, "error_start": 4, "error_end": 8,
                          "error_span": "go", "span_kind": "non_empty"}, doc)
