"""Leakage-safe S/L/P/M/D feature extraction for prepared NUCLE v2.

Run from the repository root: .venv/Scripts/python.exe src/features/nucle_features.py
The encoder is fitted on train rows only. Labels, IDs and provenance are
written to a separate targets file and never enter the predictor matrix.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

try:
    from src.data.constants import DATA_ROOT, PROJECT_ROOT
    from src.features.nucle_pos import align_span, load_tagger, require
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import DATA_ROOT, PROJECT_ROOT
    from src.features.nucle_pos import align_span, load_tagger, require

PREPARED_DIR = DATA_ROOT / "processed" / "nucle_preparation_v2"
SPLIT_DIR = DATA_ROOT / "split" / "nucle_group_split"
DEFAULT_OUTPUT = DATA_ROOT / "features" / "nucle_features_v1"
MORPH_FIELDS = ("Number", "Person", "Tense", "VerbForm")
LEXICAL_LIMIT = 500
SPECIAL = "__MISSING__"
UNK = "__UNK__"
FORBIDDEN_FEATURE_NAMES = {"error_type", "type", "correction", "corrected_sentence", "sample_id",
                           "annotation_id", "essay_id", "sentence_id", "group_id", "comment",
                           "review", "provenance", "input_sha256", "alignment"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_word(token) -> str:
    return token.text.casefold()


def _span_and_context(doc, start: int, end: int) -> tuple[list[int], list[int], list[int], dict]:
    aligned = align_span(doc, start, end)
    ids = aligned["span_token_indices"]
    if ids:
        left_edge, right_edge = ids[0], ids[-1] + 1
    elif start == end:
        left_edge = next((i for i, t in enumerate(doc) if t.idx >= start), len(doc))
        right_edge = left_edge
    else:
        left_edge = right_edge = 0
    left = list(range(max(0, left_edge - 2), left_edge))
    right = list(range(right_edge, min(len(doc), right_edge + 2)))
    return ids, left, right, aligned


def lexical_candidates(doc, ids: list[int], left: list[int], right: list[int]) -> list[tuple[str, ...]]:
    """Unigrams/bigrams in span and each contiguous two-token context region."""
    regions = [ids, left, right]
    result: list[tuple[str, ...]] = []
    for region in regions:
        words = [_token_word(doc[i]) for i in region]
        result.extend(("U", w) for w in words)
        result.extend(("B", words[i], words[i + 1]) for i in range(len(words) - 1))
    return result


def extract_features(sample: dict, doc) -> dict[str, float | str]:
    """Extract predictors using only original sentence and gold span offsets.

    Deliberately does not accept/read error_type, IDs, correction, comments,
    review flags, or preparation provenance.
    """
    sentence = sample["sentence"]
    start, end = sample["error_start"], sample["error_end"]
    require(doc.text == sentence, "Parsed document text differs from source sentence")
    require(0 <= start <= end <= len(sentence), "Span offsets outside sentence")
    require(sentence[start:end] == sample["error_span"], "Span/source text mismatch")
    ids, left, right, pos = _span_and_context(doc, start, end)
    feats: dict[str, float | str] = {
        "S|sentence_tokens": float(len(doc)),
        "S|span_tokens": float(len(ids)),
        "S|span_chars": float(end - start),
        "S|start_relative": start / max(1, len(sentence)),
        "S|end_relative": end / max(1, len(sentence)),
        "S|span_kind": sample.get("span_kind", "non_empty"),
        "P|span_signature": pos["pos_signature"],
        "P|span_alignment": pos["span_alignment"],
        "P|local_pattern": "|".join((doc[i].pos_ or SPECIAL) for i in left + ids + right) or SPECIAL,
    }
    # Position slots are explicitly represented; absence stays missing.
    slots = {"left_2": left[-2] if len(left) == 2 else None,
             "left_1": left[-1] if left else None,
             "right_1": right[0] if right else None,
             "right_2": right[1] if len(right) > 1 else None}
    for slot, index in slots.items():
        tok = doc[index] if index is not None else None
        feats[f"L|{slot}"] = _token_word(tok) if tok else SPECIAL
        feats[f"P|{slot}"] = tok.pos_ or SPECIAL if tok else SPECIAL
    feats["L|span_unigrams"] = "\x1f".join(_token_word(doc[i]) for i in ids) or SPECIAL
    feats["L|span_ngrams"] = "\x1f".join(" ".join(c[1:]) for c in lexical_candidates(doc, ids, left, right)) or SPECIAL

    for field in MORPH_FIELDS:
        values = sorted({value for i in ids for value in doc[i].morph.get(field)})
        feats[f"M|{field}"] = "+".join(values) if values else ("NOT_APPLICABLE" if not ids else SPECIAL)
    # Fixed span aggregation: ordered unique dependency/POS signatures, numeric
    # mean head distance; no governor/dependent surface forms or lemmas.
    feats["D|dep_signature"] = "+".join(dict.fromkeys(doc[i].dep_ or SPECIAL for i in ids)) if ids else "NOT_APPLICABLE"
    feats["D|head_pos"] = "+".join(dict.fromkeys(doc[i].head.pos_ or SPECIAL for i in ids)) if ids else "NOT_APPLICABLE"
    feats["D|head_dep"] = "+".join(dict.fromkeys(doc[i].head.dep_ or SPECIAL for i in ids)) if ids else "NOT_APPLICABLE"
    feats["D|head_distance"] = (sum(abs(doc[i].head.i - i) for i in ids) / len(ids)) if ids else 0.0
    subjects = [child for i in ids for child in doc[i].children
                if child.dep_.lower() in {"nsubj", "nsubjpass", "csubj", "csubjpass"}]
    feats["D|has_subject"] = float(bool(subjects))
    feats["D|subject_pos"] = "+".join(sorted({t.pos_ or SPECIAL for t in subjects})) or ("NOT_APPLICABLE" if not ids else SPECIAL)
    for field in ("Number", "Person"):
        vals = sorted({v for t in subjects for v in t.morph.get(field)})
        feats[f"D|subject_{field}"] = "+".join(vals) if vals else ("NOT_APPLICABLE" if not ids else SPECIAL)
    distances = [abs(t.i - i) for i in ids for t in subjects]
    feats["D|subject_distance"] = sum(distances) / len(distances) if distances else 0.0
    return feats


class TrainOnlyEncoder:
    """Deterministic sparse categorical encoder with train-only vocabularies."""
    def fit(self, rows: list[dict]) -> "TrainOnlyEncoder":
        require(bool(rows), "Cannot fit encoder on no rows")
        require(all(r.get("partition") == "train" for r in rows), "Encoder fit accepts train rows only")
        ids = [r["sample_id"] for r in rows]
        require(len(ids) == len(set(ids)), "Duplicate sample IDs in fit rows")
        self.train_ids = set(ids)
        term_counts: Counter[tuple[str, ...]] = Counter()
        category_values: dict[str, set[str]] = {}
        numeric_keys: set[str] = set()
        for row in rows:
            for key, value in row["features"].items():
                require(key.split("|", 1)[0] in {"S", "L", "P", "M", "D"},
                        f"Unexpected feature family: {key}")
                require(key.casefold() not in FORBIDDEN_FEATURE_NAMES,
                        f"Forbidden leakage field in predictors: {key}")
                if key == "L|span_ngrams":
                    for term in value.split("\x1f") if value != SPECIAL else ():
                        parts = term.split(" ")
                        term_counts[("B", *parts) if len(parts) > 1 else ("U", parts[0])] += 1
                elif key == "L|span_unigrams":
                    continue
                elif isinstance(value, (int, float)):
                    numeric_keys.add(key)
                else:
                    category_values.setdefault(key, set()).add(str(value))
        self.lexical_vocab = [term for term, _ in sorted(term_counts.items(), key=lambda x: (-x[1], x[0]))[:LEXICAL_LIMIT]]
        self.categories = {k: sorted(v | {SPECIAL, "NOT_APPLICABLE", UNK}) for k, v in sorted(category_values.items())}
        self.numeric_keys = sorted(numeric_keys)
        names = [f"N|{k}" for k in self.numeric_keys]
        for key, values in self.categories.items():
            names.extend(f"C|{key}|{v}" for v in values)
        for term in self.lexical_vocab:
            names.append("L|span_ngrams|" + " ".join(term[1:]))
        names.extend(["L|span_ngrams|__UNK__"])
        self.feature_names = names
        self.index = {name: i for i, name in enumerate(names)}
        return self

    def transform_one(self, features: dict) -> dict[int, float]:
        require(all(key.split("|", 1)[0] in {"S", "L", "P", "M", "D"} for key in features),
                "Unexpected feature family during transform")
        require(not any(key.casefold() in FORBIDDEN_FEATURE_NAMES for key in features),
                "Forbidden leakage field during transform")
        out: dict[int, float] = {}
        for key in self.numeric_keys:
            if key in features:
                out[self.index[f"N|{key}"]] = float(features[key])
        for key, values in self.categories.items():
            value = str(features.get(key, SPECIAL))
            if value not in values:
                value = UNK
            out[self.index[f"C|{key}|{value}"]] = 1.0
        terms = []
        value = features.get("L|span_ngrams", SPECIAL)
        if value != SPECIAL:
            for term in value.split("\x1f"):
                parts = term.split(" ")
                terms.append(("B", *parts) if len(parts) > 1 else ("U", parts[0]))
        vocab = set(self.lexical_vocab)
        unknown = False
        for term in terms:
            name = "L|span_ngrams|" + " ".join(term[1:])
            if term in vocab:
                out[self.index[name]] = 1.0
            else:
                unknown = True
        if unknown:
            out[self.index["L|span_ngrams|__UNK__"]] = 1.0
        return out


def run(prepared_dir: Path = PREPARED_DIR, split_dir: Path = SPLIT_DIR, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    samples = [json.loads(line) for line in (prepared_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with (split_dir / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        split = {r["sample_id"]: r for r in csv.DictReader(f)}
    require(len(split) == len(samples) and {r["sample_id"] for r in samples} == set(split), "Prepared/split IDs differ")
    nlp = load_tagger()
    text_by_id = {}
    for row in samples:
        sid = row["sentence_id"]
        require(sid not in text_by_id or text_by_id[sid] == row["sentence"], "Sentence ID collision")
        text_by_id[sid] = row["sentence"]
    parsed = {sid: doc for sid, doc in zip(text_by_id, nlp.pipe(text_by_id.values(), batch_size=256), strict=True)}
    feature_rows, target_rows = [], []
    for row in samples:
        sid = row["sample_id"]
        feature_rows.append({"sample_id": sid, "partition": split[sid]["partition"],
                             "features": extract_features(row, parsed[row["sentence_id"]])})
        target_rows.append({"sample_id": sid, "partition": split[sid]["partition"], "error_type": row["error_type"]})
    train = [r for r in feature_rows if r["partition"] == "train"]
    encoder = TrainOnlyEncoder().fit(train)
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with (output_dir / "features.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in feature_rows:
            json.dump({"sample_id": row["sample_id"], "partition": row["partition"],
                       "x": encoder.transform_one(row["features"])}, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    with (output_dir / "targets.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in target_rows:
            json.dump(row, f, ensure_ascii=False, sort_keys=True); f.write("\n")
    encoder_json = {"schema": "nucle-slpmd-v1", "fit_partition": "train", "train_sample_count": len(train),
                    "lexical_limit": LEXICAL_LIMIT, "lexical_vocabulary": [list(t) for t in encoder.lexical_vocab],
                    "feature_names": encoder.feature_names,
                    "train_sample_id_sha256": hashlib.sha256("\n".join(sorted(encoder.train_ids)).encode()).hexdigest()}
    (output_dir / "encoder.json").write_text(json.dumps(encoder_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "nucle-slpmd-v1", "sample_count": len(samples),
                "partition_counts": dict(Counter(r["partition"] for r in feature_rows)),
                "feature_dimension": len(encoder.feature_names), "unique_sentences_parsed": len(parsed),
                "test_scored": False, "inputs": {"prepared_manifest_sha256": (prepared_dir / "manifest.sha256").read_text(encoding="utf-8").strip(),
                                                    "split_manifest_sha256": sha256(split_dir / "split_manifest.csv")},
                "outputs": {name: sha256(output_dir / name) for name in ("features.jsonl", "targets.jsonl", "encoder.json")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.split, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
