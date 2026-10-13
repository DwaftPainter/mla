"""POS span contract and POS-only diagnostic baseline for prepared NUCLE v2.

Run from the project root with ``.venv/Scripts/python.exe src/features/nucle_pos.py`` after installing
the pinned English spaCy model. Only train labels fit the POS lookup; validation
is scored once, and test sentences are not parsed or scored by this command.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

try:
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
except ModuleNotFoundError:  # direct-file execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS


SPACY_VERSION = "3.8.7"
MODEL_NAME = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
MODEL_WHEEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
PREPARED_DIR = DATA_ROOT / "processed" / "nucle_preparation_v2"
SPLIT_DIR = DATA_ROOT / "split" / "nucle_group_split"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "nucle_pos_baseline_v1"
NA = "NOT_APPLICABLE"
MISSING = "POS_MISSING"
LABEL_ORDER = tuple(TARGET_LABELS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_tagger():
    import spacy

    require(spacy.__version__ == SPACY_VERSION,
            f"Expected spaCy {SPACY_VERSION}, got {spacy.__version__}")
    try:
        nlp = spacy.load(MODEL_NAME, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"Missing {MODEL_NAME} {MODEL_VERSION}. Install the pinned wheel: {MODEL_URL}"
        ) from exc
    meta = nlp.meta
    require(meta.get("name") == "core_web_sm" and meta.get("version") == MODEL_VERSION,
            f"Expected {MODEL_NAME} {MODEL_VERSION}, got {meta.get('name')} {meta.get('version')}")
    require("tagger" in nlp.pipe_names and "parser" in nlp.pipe_names,
            "The selected English model must provide a trained tagger and parser")
    return nlp


def align_span(doc, start: int, end: int) -> dict:
    """Map a sentence-relative half-open character span to spaCy token POS.

    Non-empty spans use every token with character overlap. Partial-token spans
    therefore keep all intersecting tokens and are flagged explicitly. Leading
    or trailing whitespace is ignored only for token selection and recorded as
    ``whitespace_edge``. Insertions have no target POS; boundary neighbors are
    retained as context, while insertions inside a token are marked separately.
    """
    text = doc.text
    require(isinstance(start, int) and isinstance(end, int), "Span offsets must be integers")
    require(0 <= start <= end <= len(text), "Span offsets are outside the sentence")

    tokens = list(doc)
    if start == end:
        left = [i for i, token in enumerate(tokens) if token.idx + len(token) <= start]
        right = [i for i, token in enumerate(tokens) if token.idx >= start]
        inside = [i for i, token in enumerate(tokens) if token.idx < start < token.idx + len(token)]
        return {
            "span_alignment": "insertion_inside_token" if inside else "insertion_boundary",
            "span_token_indices": [],
            "span_upos": [],
            "pos_signature": NA,
            "primary_upos": NA,
            "left_upos": tokens[left[-1]].pos_ if left else MISSING,
            "right_upos": tokens[right[0]].pos_ if right else MISSING,
            "insertion_enclosing_upos": tokens[inside[0]].pos_ if inside else NA,
            "pos_status": "not_applicable",
        }

    selected = [
        (i, token) for i, token in enumerate(tokens)
        if token.idx < end and token.idx + len(token) > start
    ]
    if not selected:
        return {
            "span_alignment": "whitespace_only", "span_token_indices": [],
            "span_upos": [], "pos_signature": MISSING, "primary_upos": MISSING,
            "left_upos": MISSING, "right_upos": MISSING,
            "insertion_enclosing_upos": NA, "pos_status": "missing",
        }

    first_i, first = selected[0]
    last_i, last = selected[-1]
    cuts_token = start > first.idx or end < last.idx + len(last)
    whitespace_edge = (
        not cuts_token
        and ((start < first.idx and text[start:first.idx].isspace())
             or (end > last.idx + len(last) and text[last.idx + len(last):end].isspace()))
    )
    alignment = "partial_token" if cuts_token else "whitespace_edge" if whitespace_edge else "exact"
    upos = [token.pos_ or MISSING for _, token in selected]
    if any(tag == MISSING for tag in upos):
        pos_status = "missing"
    else:
        pos_status = "ok"
    return {
        "span_alignment": alignment,
        "span_token_indices": [i for i, _ in selected],
        "span_upos": upos,
        "pos_signature": "+".join(upos) if pos_status == "ok" else MISSING,
        "primary_upos": upos[0] if len(upos) == 1 and pos_status == "ok" else NA,
        "left_upos": tokens[first_i - 1].pos_ or MISSING if first_i > 0 else MISSING,
        "right_upos": tokens[last_i + 1].pos_ or MISSING if last_i + 1 < len(tokens) else MISSING,
        "insertion_enclosing_upos": NA,
        "pos_status": pos_status,
    }


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict, list[list[int]]]:
    matrix = [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]
    index = {label: i for i, label in enumerate(LABEL_ORDER)}
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[index[truth]][index[prediction]] += 1
    per_label = {}
    scores = []
    for i, label in enumerate(LABEL_ORDER):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return sum(scores) / len(scores), per_label, matrix


class POSRule:
    """ADP/DET anchors; other observed signatures use train-only majority."""

    FIXED = {"ADP": "Prep", "DET": "ArtOrDet"}

    def fit(self, rows: list[dict]) -> "POSRule":
        self.global_counts = Counter(row["error_type"] for row in rows)
        require(bool(self.global_counts), "Cannot fit POS rule without training rows")
        self.signature_counts: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            signature = row["pos_signature"]
            if signature not in {NA, MISSING}:
                self.signature_counts[signature][row["error_type"]] += 1
        self.majority = self._winner(self.global_counts)
        return self

    @staticmethod
    def _winner(counts: Counter) -> str:
        return min(LABEL_ORDER, key=lambda label: (-counts.get(label, 0), LABEL_ORDER.index(label)))

    def predict(self, row: dict) -> str:
        signature = row["pos_signature"]
        if signature in self.FIXED:
            return self.FIXED[signature]
        counts = self.signature_counts.get(signature)
        return self._winner(counts) if counts else self.majority

    def table(self) -> list[dict]:
        return [
            {"pos_signature": signature, "support": sum(counts.values()),
             **{label: counts.get(label, 0) for label in LABEL_ORDER},
             "rule": "fixed_ADP_to_Prep" if signature == "ADP" else
                     "fixed_DET_to_ArtOrDet" if signature == "DET" else "train_majority"}
            for signature, counts in sorted(self.signature_counts.items())
        ]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(prepared_dir: Path = PREPARED_DIR, split_dir: Path = SPLIT_DIR,
        output_dir: Path = DEFAULT_OUTPUT) -> dict:
    samples = read_jsonl(prepared_dir / "samples.jsonl")
    split_path = split_dir / "split_manifest.csv"
    partitions = {}
    with split_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            partitions[row["sample_id"]] = row["partition"]
    require(len(partitions) == len(set(partitions)), "Duplicate sample IDs in split manifest")
    require({row["sample_id"] for row in samples} == set(partitions),
            "Prepared samples and locked split IDs differ")

    # Do not parse or score the held-out test partition in this diagnostic run.
    working = [row for row in samples if partitions[row["sample_id"]] in {"train", "validation"}]
    try:
        nlp = load_tagger()
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before running POS extraction") from exc
    sentence_text = {}
    for row in working:
        sentence_text.setdefault(row["sentence_id"], row["sentence"])
        require(sentence_text[row["sentence_id"]] == row["sentence"],
                "Sentence ID maps to different sentence text")
    parsed = {sid: nlp(text) for sid, text in sentence_text.items()}

    rows = []
    for sample in working:
        doc = parsed[sample["sentence_id"]]
        start, end = sample["error_start"], sample["error_end"]
        require(doc.text[start:end] == sample["error_span"],
                f"Sample span/source mismatch for annotation {sample['annotation_id']}")
        aligned = align_span(doc, start, end)
        rows.append({**aligned, "sample_id": sample["sample_id"],
                     "annotation_id": sample["annotation_id"], "essay_id": sample["essay_id"],
                     "error_type": sample["error_type"], "partition": partitions[sample["sample_id"]],
                     "preparation_alignment": sample["alignment"]})

    train_rows = [row for row in rows if row["partition"] == "train"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    rule = POSRule().fit(train_rows)
    predictions = [rule.predict(row) for row in validation_rows]
    truths = [row["error_type"] for row in validation_rows]
    score, per_label, matrix = macro_f1(truths, predictions)
    majority = POSRule()
    majority.global_counts = Counter(row["error_type"] for row in train_rows)
    majority.majority = majority._winner(majority.global_counts)
    majority_predictions = [majority.majority] * len(validation_rows)
    majority_score, majority_per_label, majority_matrix = macro_f1(truths, majority_predictions)

    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"Output already exists; choose a new directory: {output_dir}")
    output_dir.mkdir(parents=True)
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "sample_id", "annotation_id", "essay_id", "error_type", "prediction",
            "pos_signature", "span_alignment", "preparation_alignment", "pos_status"])
        writer.writeheader()
        for row, prediction in zip(validation_rows, predictions, strict=True):
            writer.writerow({**{key: row[key] for key in writer.fieldnames if key in row},
                             "prediction": prediction})
    with (output_dir / "pos_by_label_train.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["pos_signature", "support", *LABEL_ORDER, "rule"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rule.table())

    summary = {
        "pos_contract": "nucle-pos-span-v1",
        "spacy_version": SPACY_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_wheel_sha256": MODEL_WHEEL_SHA256,
        "model_components": [component for component in nlp.pipe_names if component != "ner"],
        "model_meta_sha256": hashlib.sha256(json.dumps(nlp.meta, sort_keys=True).encode()).hexdigest(),
        "python_version": platform.python_version(),
        "source": "original sentence and character offsets only",
        "test_partition_parsed_or_scored": False,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "pos_missing_or_na_validation": sum(row["pos_status"] != "ok" for row in validation_rows),
        "validation_alignment_counts": dict(Counter(row["span_alignment"] for row in validation_rows)),
        "preparation_alignment_counts_validation": dict(Counter(row["preparation_alignment"] for row in validation_rows)),
        "pos_rule": {
            "single_ADP": "Prep", "single_DET": "ArtOrDet",
            "other_observed_POS_signature": "training majority for the ordered signature",
            "unseen_signature_or_missing_or_insertion": f"training majority ({rule.majority})",
            "tie_break": "fixed label order: " + ", ".join(LABEL_ORDER),
            "fit_source": "train partition only",
        },
        "majority_baseline_validation": {
            "majority_label": majority.majority, "macro_f1": majority_score,
            "per_label": majority_per_label, "confusion_matrix_true_rows_predicted_columns": majority_matrix,
        },
        "pos_rule_validation": {
            "macro_f1": score, "per_label": per_label,
            "confusion_matrix_true_rows_predicted_columns": matrix,
        },
        "input_hashes": {
            "prepared_manifest_sha256": (prepared_dir / "manifest.sha256").read_text(encoding="utf-8").strip(),
            "split_manifest_sha256": sha256_file(split_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.split, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
