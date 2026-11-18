"""Validation-only slice analysis from saved predictions; no model fitting.

Run from the repository root:
    python src/evaluation/nucle_span_analysis.py

Reports contain aggregate counts only. Hard pairs are gold-label slices, not
parser-confirmed verb spans or noun phrases. Predictions remain five-class.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("ArtOrDet", "Prep", "Nn", "Vform", "SVA")
PAIRS = {"Vform/SVA": ("Vform", "SVA"), "Nn/ArtOrDet": ("Nn", "ArtOrDet")}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(ids, samples, predictions, labels):
    if not ids:
        return {"support": 0, "macro_f1": None, "accuracy": None,
                "outside_label_predictions": 0, "per_label": {}}
    result = {}
    for label in labels:
        tp = sum(samples[sid]["error_type"] == label == predictions[sid] for sid in ids)
        actual = sum(samples[sid]["error_type"] == label for sid in ids)
        predicted = sum(predictions[sid] == label for sid in ids)
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        f1 = 2 * tp / (actual + predicted) if actual + predicted else 0.0
        result[label] = {"support": actual, "precision": p, "recall": r, "f1": f1}
    return {
        "support": len(ids),
        "macro_f1": sum(row["f1"] for row in result.values()) / len(labels),
        "accuracy": sum(samples[sid]["error_type"] == predictions[sid] for sid in ids) / len(ids),
        "outside_label_predictions": sum(predictions[sid] not in labels for sid in ids),
        "per_label": result,
    }


def run(output):
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    prep = ROOT / "data_set/processed/nucle_preparation_v2"
    split_dir = ROOT / "data_set/split/nucle_group_split"
    pos_dir = ROOT / "reports/nucle_pos_baseline_v1"
    ab_dir = ROOT / "reports/nucle_feature_ablation_v1"
    paths = {
        "samples": prep / "samples.jsonl", "split": split_dir / "split_manifest.csv",
        "pos_predictions": pos_dir / "validation_predictions.csv",
        "ablation_predictions": ab_dir / "validation_predictions.csv",
        "pos_summary": pos_dir / "summary.json", "ablation_summary": ab_dir / "summary.json",
        "split_summary": split_dir / "split_summary.json",
        "essay_groups": split_dir / "essay_groups.csv",
    }
    split_rows = read_csv(paths["split"])
    split = {r["sample_id"]: r["partition"] for r in split_rows}
    assert len(split) == len(split_rows)
    samples = {}
    # Read only prepared metadata into memory; never retain or output source text.
    with paths["samples"].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            assert sid not in samples
            empty = row["error_start"] == row["error_end"]
            count = len(row["span_token_indices"])
            assert empty or count > 0
            samples[sid] = {
                "error_type": row["error_type"], "tokens": count,
                "span_bin": "empty" if empty else "single_token" if count == 1 else "multi_token",
            }
    assert samples.keys() == split.keys()
    val_ids = sorted(sid for sid in split if split[sid] == "validation")
    pos_rows = read_csv(paths["pos_predictions"])
    assert len(pos_rows) == len(val_ids)
    predictions = {"POS rule": {r["sample_id"]: r["prediction"] for r in pos_rows}}
    token_count_disagreements = 0
    for row in pos_rows:
        assert split[row["sample_id"]] == "validation"
        assert row["error_type"] == samples[row["sample_id"]]["error_type"]
        if row["pos_status"] == "ok":
            token_count_disagreements += (
                len(row["pos_signature"].split("+")) != samples[row["sample_id"]]["tokens"])
    ab_rows = read_csv(paths["ablation_predictions"])
    for row in ab_rows:
        assert split[row["sample_id"]] == "validation"
        assert row["error_type"] == samples[row["sample_id"]]["error_type"]
        if row["model"] != "LogisticRegression":
            continue
        model = "LR " + row["variant"]
        pred = predictions.setdefault(model, {})
        assert row["sample_id"] not in pred
        pred[row["sample_id"]] = row["prediction"]
    for pred in predictions.values():
        assert set(pred) == set(val_ids)
        assert set(pred.values()) <= set(LABELS)
    pos_summary = read_json(paths["pos_summary"])
    ab_summary = read_json(paths["ablation_summary"])
    assert pos_summary["input_hashes"]["split_manifest_sha256"] == digest(paths["split"])
    assert ab_summary["split_manifest_sha256"] == digest(paths["split"])
    # Verify the saved predictions reproduce the published global scores.
    for model, pred in predictions.items():
        got = metric(val_ids, samples, pred, LABELS)["macro_f1"]
        expected = (pos_summary["pos_rule_validation"]["macro_f1"] if model == "POS rule"
                    else ab_summary["selected_by_validation"][model[3:]]["LogisticRegression"]["validation_macro_f1"])
        assert abs(got - expected) < 1e-10, (model, got, expected)
    bins = ("empty", "single_token", "multi_token")
    counts = {}
    for part in ("train", "validation", "test"):
        ids = [sid for sid in split if split[sid] == part]
        counts[part] = {b: sum(samples[sid]["span_bin"] == b for sid in ids) for b in bins}
    span_results = {}
    for b in bins:
        ids = [sid for sid in val_ids if samples[sid]["span_bin"] == b]
        span_results[b] = {m: metric(ids, samples, p, LABELS) for m, p in predictions.items()}
    hard_results = {}
    for name, labels in PAIRS.items():
        ids = [sid for sid in val_ids if samples[sid]["error_type"] in labels]
        hard_results[name] = {m: metric(ids, samples, p, labels) for m, p in predictions.items()}
    nn_errors = {}
    for model in ("POS rule", "LR S+L+P+M+D"):
        ids = [sid for sid in val_ids if samples[sid]["error_type"] == "Nn"
               and predictions[model][sid] == "ArtOrDet"]
        nn_errors[model] = {b: sum(samples[sid]["span_bin"] == b for sid in ids) for b in bins}
    # Label slices are not guaranteed to be POS/constituent slices.
    pair_pos = {}
    shared_pos_results = {}
    for name, labels in PAIRS.items():
        signatures = Counter(r["pos_signature"] for r in pos_rows if r["error_type"] in labels)
        pair_pos[name] = dict(sorted(signatures.items(), key=lambda item: (-item[1], item[0])))
        shared_pos_results[name] = {}
        for signature in ("VERB", "AUX", "NOUN", "DET"):
            ids = [r["sample_id"] for r in pos_rows
                   if r["error_type"] in labels and r["pos_signature"] == signature]
            shared_pos_results[name][signature] = {
                m: metric(ids, samples, pred, labels) for m, pred in predictions.items()
            }
    essays = read_csv(paths["essay_groups"])
    all_essay_split_counts = {}
    for part in ("train", "validation", "test"):
        rows = [r for r in essays if r["partition"] == part]
        all_essay_split_counts[part] = {
            "essays": len(rows), "groups": len({r["group_id"] for r in rows})}
    report = {
        "scope": "saved validation predictions only; no fitting or test scoring",
        "span_definition": "prepared token overlap count; empty iff character start=end",
        "hard_subset_definition": "gold-label pair; original five-class predictions kept, including outside-pair errors",
        "span_macro_f1_labels": list(LABELS),
        "hard_macro_f1_labels": {k: list(v) for k, v in PAIRS.items()},
        "token_count_disagreements_prepared_vs_validation_pos": token_count_disagreements,
        "span_counts": counts, "span_validation": span_results,
        "hard_pair_validation": hard_results, "Nn_to_ArtOrDet_by_span": nn_errors,
        "hard_pair_pos_signatures": pair_pos,
        "hard_pair_fixed_single_pos_validation": shared_pos_results,
        "all_essay_split_counts": all_essay_split_counts,
        "split_summary": read_json(paths["split_summary"]),
        "parser": {k: pos_summary[k] for k in ("spacy_version", "model", "model_version", "model_components")},
        "test_scored": False,
        "input_sha256": {k: digest(p) for k, p in paths.items()},
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for scope, groups in (("span", span_results), ("hard_pair", hard_results)):
        for subset, models in groups.items():
            for model, result in models.items():
                rows.append({"scope": scope, "subset": subset, "model": model,
                             **{k: result[k] for k in ("support", "macro_f1", "accuracy", "outside_label_predictions")}})
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/nucle_span_diagnostics_v1")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ("span_counts", "Nn_to_ArtOrDet_by_span",
          "token_count_disagreements_prepared_vs_validation_pos")}, indent=2))
    for subset, models in result["hard_pair_validation"].items():
        for name in ("POS rule", "LR S+L+P+M+D"):
            print(subset, name, models[name])
