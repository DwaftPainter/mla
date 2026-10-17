"""Run train-fit/validation-selection ablations for NUCLE S/L/P/M/D features.

Run from the repository root:
    .venv/Scripts/python.exe src/training/nucle_ablation.py

The locked test partition is never loaded into model inputs or scored.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch

try:
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
    from src.evaluation.metrics import accuracy, scores
    from src.models.nucle_logistic import (
        LR_MAX_EVAL, LR_MAX_ITER, LR_TOLERANCE_CHANGE, LR_TOLERANCE_GRAD,
        fit_logistic,
    )
    from src.models.nucle_tree import GiniTree
    from src.features.nucle_features import (
        PREPARED_DIR, SPECIAL, TrainOnlyEncoder, _span_and_context,
        extract_features, lexical_candidates,
    )
    from src.features.nucle_pos import load_tagger
    from src.training.nucle_benchmark import load_rows, sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
    from src.evaluation.metrics import accuracy, scores
    from src.models.nucle_logistic import (
        LR_MAX_EVAL, LR_MAX_ITER, LR_TOLERANCE_CHANGE, LR_TOLERANCE_GRAD,
        fit_logistic,
    )
    from src.models.nucle_tree import GiniTree
    from src.features.nucle_features import (
        PREPARED_DIR, SPECIAL, TrainOnlyEncoder, _span_and_context,
        extract_features, lexical_candidates,
    )
    from src.features.nucle_pos import load_tagger
    from src.training.nucle_benchmark import load_rows, sha256

FEATURE_DIR = DATA_ROOT / "features" / "nucle_features_v1"
SPLIT_DIR = DATA_ROOT / "split" / "nucle_group_split"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "nucle_feature_ablation_v1"
LABELS = tuple(TARGET_LABELS)

# The first four are the preregistered incremental sequence. The last two
# isolate direct lexical identity in the target span and all lexical features.
VARIANTS = (
    ("S+L", {"S", "L"}, "Incremental baseline"),
    ("S+L+P", {"S", "L", "P"}, "Add POS"),
    ("S+L+P+M", {"S", "L", "P", "M"}, "Add morphology"),
    ("S+L+P+M+D", {"S", "L", "P", "M", "D"}, "Full features"),
    ("Full_mask_span_lexical", {"S", "L", "P", "M", "D"}, "Full; mask target-span lexical identity"),
    ("S+P+M+D_nonlexical", {"S", "P", "M", "D"}, "Remove all lexical features"),
)


def feature_group(name: str) -> str:
    """Return the S/L/P/M/D family for encoded and numeric feature names."""
    bits = name.split("|")
    if bits[0] == "N":
        return bits[1]
    if bits[0] == "C":
        return bits[1]
    if bits[0] == "L":
        return "L"
    raise ValueError(f"Unrecognized encoded feature name: {name}")


def feature_indices(feature_names: list[str], variant_name: str, groups: set[str]) -> list[int]:
    keep = []
    for index, name in enumerate(feature_names):
        group = feature_group(name)
        if group not in groups:
            continue
        keep.append(index)
    if not keep:
        raise ValueError(f"Variant {variant_name} selected no features")
    return keep


def make_masked_span_features(prepared_dir: Path, split_dir: Path):
    """Re-extract the mask variant before fitting its train-only vocabulary.

    Target-span lexical fields are blanked before vocabulary learning. Lexical
    unigrams/bigrams from the two-token left/right contexts are retained.
    Test sentences are not parsed or retained in this variant.
    """
    with (split_dir / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        partitions = {row["sample_id"]: row["partition"] for row in csv.DictReader(stream)}
    samples = [json.loads(line) for line in (prepared_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
               if line and partitions.get(json.loads(line)["sample_id"]) in {"train", "validation"}]
    text_by_id = {}
    for row in samples:
        sid = row["sentence_id"]
        if sid in text_by_id and text_by_id[sid] != row["sentence"]:
            raise ValueError(f"Sentence ID collision: {sid}")
        text_by_id[sid] = row["sentence"]
    nlp = load_tagger()
    parsed = {sid: doc for sid, doc in zip(text_by_id, nlp.pipe(text_by_id.values(), batch_size=256), strict=True)}
    rows = []
    for sample in samples:
        doc = parsed[sample["sentence_id"]]
        features = extract_features(sample, doc)
        _, left, right, _ = _span_and_context(doc, sample["error_start"], sample["error_end"])
        context_terms = lexical_candidates(doc, [], left, right)
        features["L|span_unigrams"] = SPECIAL
        features["L|span_ngrams"] = "\x1f".join(" ".join(term[1:]) for term in context_terms) or SPECIAL
        sid = sample["sample_id"]
        rows.append({"sample_id": sid, "partition": partitions[sid], "features": features})
    train_rows = [row for row in rows if row["partition"] == "train"]
    encoder = TrainOnlyEncoder().fit(train_rows)
    transformed = {row["sample_id"]: encoder.transform_one(row["features"]) for row in rows}
    return transformed, encoder.feature_names


def remap_rows(ids: list[str], raw: dict[str, dict[int, float]], keep: list[int]) -> list[dict[int, float]]:
    remap = {old: new for new, old in enumerate(keep)}
    return [{remap[index]: value for index, value in raw[sid].items() if index in remap and value}
            for sid in ids]


def sparse_tensor(rows: list[dict[int, float]], dimension: int) -> torch.Tensor:
    ri, ci, values = [], [], []
    for row_index, row in enumerate(rows):
        for col_index, value in row.items():
            ri.append(row_index); ci.append(col_index); values.append(value)
    indices = torch.tensor([ri, ci], dtype=torch.long)
    vals = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, (len(rows), dimension)).coalesce()


def run(feature_dir: Path = FEATURE_DIR, split_dir: Path = SPLIT_DIR,
        output_dir: Path = DEFAULT_OUTPUT, variant_names: set[str] | None = None) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    wall_clock = perf_counter()
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    encoder = json.loads((feature_dir / "encoder.json").read_text(encoding="utf-8"))
    if manifest["schema"] != encoder["schema"] or encoder["fit_partition"] != "train":
        raise ValueError("Feature schema/encoder partition contract failed")
    raw, targets, split, train_ids, val_ids = load_rows(feature_dir, split_dir)
    feature_names = encoder["feature_names"]
    y_train_labels = [targets[sid]["error_type"] for sid in train_ids]
    y_train = [LABELS.index(label) for label in y_train_labels]
    y_val = [targets[sid]["error_type"] for sid in val_ids]
    results: list[dict[str, Any]] = []
    validation_predictions: list[dict[str, str]] = []
    feature_counts: dict[str, int] = {}
    run_started = perf_counter()

    chosen_variants = [item for item in VARIANTS if variant_names is None or item[0] in variant_names]
    if not chosen_variants or (variant_names is not None and variant_names != {item[0] for item in chosen_variants}):
        raise ValueError(f"Unknown/empty variant selection: {variant_names}")
    for variant_name, groups, description in chosen_variants:
        if variant_name == "Full_mask_span_lexical":
            variant_raw, variant_names_list = make_masked_span_features(PREPARED_DIR, split_dir)
            keep = list(range(len(variant_names_list)))
            names_for_variant = variant_names_list
        else:
            variant_raw, names_for_variant = raw, feature_names
            keep = feature_indices(names_for_variant, variant_name, groups)
        feature_counts[variant_name] = len(keep)
        x_train_rows = remap_rows(train_ids, variant_raw, keep)
        x_val_rows = remap_rows(val_ids, variant_raw, keep)
        x_train = sparse_tensor(x_train_rows, len(keep))
        x_val = sparse_tensor(x_val_rows, len(keep))
        numeric_local = {new for new, old in enumerate(keep) if names_for_variant[old].startswith("N|")}
        variant_predictions: dict[tuple[Any, ...], list[str]] = {}

        for c_value in (0.01, 0.1, 1.0, 10.0):
            for balanced in (False, True):
                fit_clock = perf_counter()
                w, b, convergence = fit_logistic(x_train, y_train, c_value, balanced)
                fit_seconds = perf_counter() - fit_clock
                with torch.no_grad():
                    train_pred_idx = (torch.sparse.mm(x_train, w) + b).argmax(dim=1).tolist()
                    val_pred_idx = (torch.sparse.mm(x_val, w) + b).argmax(dim=1).tolist()
                train_pred = [LABELS[i] for i in train_pred_idx]
                pred = [LABELS[i] for i in val_pred_idx]
                variant_predictions[("LogisticRegression", c_value, balanced)] = pred
                results.append({"variant": variant_name, "description": description,
                                "model": "LogisticRegression", "configuration": {
                                    "C": c_value, "class_weight": "balanced" if balanced else None},
                                "feature_count": len(keep), "fit_seconds": fit_seconds,
                                "convergence": convergence,
                                "train_accuracy": accuracy(y_train_labels, train_pred),
                                **scores(y_val, pred)})

        for depth in (4, 8, 12):
            for leaf in (5, 20):
                for balanced in (False, True):
                    fit_clock = perf_counter()
                    tree = GiniTree(depth, leaf, balanced, numeric_local).fit(x_train_rows, y_train)
                    fit_seconds = perf_counter() - fit_clock
                    train_pred = [LABELS[i] for i in tree.predict(x_train_rows)]
                    pred = [LABELS[i] for i in tree.predict(x_val_rows)]
                    variant_predictions[("DecisionTree", depth, leaf, balanced)] = pred
                    results.append({"variant": variant_name, "description": description,
                                    "model": "DecisionTree", "configuration": {
                                        "max_depth": depth, "min_samples_leaf": leaf,
                                        "class_weight": "balanced" if balanced else None},
                                    "feature_count": len(keep), "fit_seconds": fit_seconds,
                                    "train_accuracy": accuracy(y_train_labels, train_pred),
                                    **scores(y_val, pred)})

        for model_name in ("LogisticRegression", "DecisionTree"):
            candidates = [row for row in results if row["variant"] == variant_name and row["model"] == model_name]
            eligible = [row for row in candidates if model_name != "LogisticRegression"
                        or row["convergence"]["converged"]]
            # max() preserves the preregistered grid's first candidate on ties.
            best = max(eligible, key=lambda row: row["macro_f1"])
            config = best["configuration"]
            if model_name == "LogisticRegression":
                key = (model_name, config["C"], config["class_weight"] == "balanced")
            else:
                key = (model_name, config["max_depth"], config["min_samples_leaf"],
                       config["class_weight"] == "balanced")
            for sid, truth, prediction in zip(val_ids, y_val, variant_predictions[key], strict=True):
                validation_predictions.append({"sample_id": sid, "error_type": truth,
                                               "variant": variant_name, "model": model_name,
                                               "prediction": prediction})

    selected = {}
    for variant_name, _, _ in chosen_variants:
        selected[variant_name] = {}
        for model_name in ("LogisticRegression", "DecisionTree"):
            candidates = [row for row in results if row["variant"] == variant_name and row["model"] == model_name]
            eligible = [row for row in candidates if model_name != "LogisticRegression"
                        or row["convergence"]["converged"]]
            best = max(eligible, key=lambda row: row["macro_f1"])
            selected[variant_name][model_name] = {
                "configuration": best["configuration"], "validation_macro_f1": best["macro_f1"],
                "validation_accuracy": best["accuracy"], "train_accuracy": best["train_accuracy"],
                "converged": best.get("convergence", {}).get("converged", True),
            }

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report = {
        "experiment": "NUCLE S/L/P/M/D incremental ablation and lexical masking",
        "feature_schema": manifest["schema"],
        "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
        "split_manifest_sha256": sha256(split_dir / "split_manifest.csv"),
        "fit_partition": "train", "selection_partition": "validation", "test_scored": False,
        "train_samples": len(train_ids), "validation_samples": len(val_ids),
        "variant_feature_counts": feature_counts,
        "logistic_solver": {"optimizer": "PyTorch LBFGS", "max_iter": LR_MAX_ITER,
                            "max_eval": LR_MAX_EVAL, "tolerance_grad": LR_TOLERANCE_GRAD,
                            "tolerance_change": LR_TOLERANCE_CHANGE},
        "selected_by_validation": selected,
        "fit_count": len(results),
        "runtime_seconds": perf_counter() - run_started,
        "started_at_utc": started_at.isoformat(timespec="milliseconds"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "error_type", "variant", "model", "prediction"])
        writer.writeheader(); writer.writerows(validation_predictions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=[item[0] for item in VARIANTS], action="append",
                        help="Run one variant (repeatable); default runs all six")
    args = parser.parse_args()
    report = run(args.features, args.split, args.output, set(args.variant) if args.variant else None)
    print(json.dumps({"output": str(args.output), "fit_count": report["fit_count"],
                      "runtime_seconds": report["runtime_seconds"],
                      "selected_by_validation": report["selected_by_validation"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
