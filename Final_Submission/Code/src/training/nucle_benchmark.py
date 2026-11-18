"""Train sparse Logistic Regression and Decision Tree on S/L/P/M/D features.

Run from the repository root after feature extraction:
    .venv/Scripts/python.exe src/training/nucle_benchmark.py

Only train labels fit parameters; validation Macro-F1 selects a configuration.
The test partition is excluded from fitting, selection, and reported metrics.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

try:
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
    from src.evaluation.metrics import accuracy, scores
    from src.models.nucle_logistic import (
        LR_MAX_EVAL, LR_MAX_ITER, LR_TOLERANCE_CHANGE, LR_TOLERANCE_GRAD,
        fit_logistic,
    )
    from src.models.nucle_tree import GiniTree
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
    from src.evaluation.metrics import accuracy, scores
    from src.models.nucle_logistic import (
        LR_MAX_EVAL, LR_MAX_ITER, LR_TOLERANCE_CHANGE, LR_TOLERANCE_GRAD,
        fit_logistic,
    )
    from src.models.nucle_tree import GiniTree

FEATURE_DIR = DATA_ROOT / "features" / "nucle_features_v1"
SPLIT_DIR = DATA_ROOT / "split" / "nucle_group_split"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "nucle_feature_benchmark_solver_v2"
LABELS = tuple(TARGET_LABELS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(feature_dir: Path, split_dir: Path):
    with (split_dir / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        split_rows = {r["sample_id"]: r for r in csv.DictReader(f)}
    features, targets = {}, {}
    feature_ids, target_ids = set(), set()
    with (feature_dir / "features.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            require(sid not in feature_ids, f"Duplicate feature row: {sid}")
            feature_ids.add(sid)
            require(row["partition"] == split_rows[sid]["partition"], f"Feature partition mismatch for {sid}")
            if row["partition"] in {"train", "validation"}:
                features[sid] = {int(k): float(v) for k, v in row["x"].items()}
    with (feature_dir / "targets.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            require(sid not in target_ids, f"Duplicate target row: {sid}")
            target_ids.add(sid)
            if row["partition"] in {"train", "validation"}:
                targets[sid] = row
    require(feature_ids == target_ids == split_rows.keys(), "Feature, target, and locked split IDs differ")
    for sid, target in targets.items():
        require(target["partition"] == split_rows[sid]["partition"], f"Partition mismatch for {sid}")
        require(target["error_type"] in LABELS, f"Unexpected target label for {sid}")
    train_ids = [sid for sid in split_rows if split_rows[sid]["partition"] == "train"]
    val_ids = [sid for sid in split_rows if split_rows[sid]["partition"] == "validation"]
    # Test rows are only skipped after reading each line's ID/partition; their
    # vectors and labels are not retained, fit, scored, or written to results.
    return features, targets, split_rows, train_ids, val_ids


def matrix_for(ids: list[str], features: dict[str, dict[int, float]], dimension: int):
    row_idx, col_idx, values = [], [], []
    for i, sid in enumerate(ids):
        for j, value in features[sid].items():
            require(0 <= j < dimension, f"Feature index outside encoder dimension: {j}")
            if value:
                row_idx.append(i); col_idx.append(j); values.append(value)
    indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
    vals = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, (len(ids), dimension)).coalesce()


def run(feature_dir: Path = FEATURE_DIR, split_dir: Path = SPLIT_DIR, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    run_started_at = datetime.now(timezone.utc)
    run_clock = perf_counter()
    data_clock = perf_counter()
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    encoder = json.loads((feature_dir / "encoder.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == encoder["schema"] == "nucle-slpmd-v1", "Feature schema mismatch")
    require(encoder["fit_partition"] == "train", "Encoder was not fit on train")
    features, targets, split, train_ids, val_ids = load_rows(feature_dir, split_dir)
    y_train = [LABELS.index(targets[sid]["error_type"]) for sid in train_ids]
    y_val = [targets[sid]["error_type"] for sid in val_ids]
    dimension = len(encoder["feature_names"])
    x_train = matrix_for(train_ids, features, dimension)
    x_val = matrix_for(val_ids, features, dimension)
    sparse_train = [features[sid] for sid in train_ids]
    sparse_val = [features[sid] for sid in val_ids]
    numeric_features = {i for i, name in enumerate(encoder["feature_names"]) if name.startswith("N|")}
    data_loading_seconds = perf_counter() - data_clock
    results, validation_predictions = [], {}
    y_train_labels = [LABELS[i] for i in y_train]
    majority_label = min(LABELS, key=lambda label: (-y_train_labels.count(label), LABELS.index(label)))
    results.append({"model": "Majority", "configuration": {"label": majority_label},
                    "train_accuracy": accuracy(y_train_labels, [majority_label] * len(y_train_labels)),
                    **scores(y_val, [majority_label] * len(y_val))})

    for c_value in (0.01, 0.1, 1.0, 10.0):
        for balanced in (False, True):
            fit_clock = perf_counter()
            w, b, convergence = fit_logistic(x_train, y_train, c_value, balanced)
            fit_seconds = perf_counter() - fit_clock
            with torch.no_grad():
                train_idx = (torch.sparse.mm(x_train, w) + b).argmax(dim=1).tolist()
                val_idx = (torch.sparse.mm(x_val, w) + b).argmax(dim=1).tolist()
            pred = [LABELS[i] for i in val_idx]
            train_pred = [LABELS[i] for i in train_idx]
            key = ("LogisticRegression", c_value, balanced)
            validation_predictions[key] = pred
            results.append({"model": "LogisticRegression", "configuration": {"C": c_value,
                            "class_weight": "balanced" if balanced else None},
                           "fit_seconds": fit_seconds, "convergence": convergence,
                           "train_accuracy": accuracy(y_train_labels, train_pred),
                           **scores(y_val, pred)})

    for depth in (4, 8, 12):
        for leaf in (5, 20):
            for balanced in (False, True):
                fit_clock = perf_counter()
                tree = GiniTree(depth, leaf, balanced, numeric_features).fit(sparse_train, y_train)
                fit_seconds = perf_counter() - fit_clock
                pred = [LABELS[i] for i in tree.predict(sparse_val)]
                train_pred = [LABELS[i] for i in tree.predict(sparse_train)]
                validation_predictions[("DecisionTree", depth, leaf, balanced)] = pred
                results.append({"model": "DecisionTree", "configuration": {"max_depth": depth,
                                "min_samples_leaf": leaf, "class_weight": "balanced" if balanced else None},
                               "fit_seconds": fit_seconds,
                               "train_accuracy": accuracy(y_train_labels, train_pred), **scores(y_val, pred)})

    selected = {}
    for model_name in ("LogisticRegression", "DecisionTree"):
        candidates = [r for r in results if r["model"] == model_name]
        # Fixed tie order: LR C ascending, then class_weight None/balanced;
        # Tree depth ascending, leaf size ascending, then None/balanced.
        # A non-converged LR candidate is reported but cannot be selected.
        eligible = [r for r in candidates if r["model"] != "LogisticRegression" or r["convergence"]["converged"]]
        best = max(eligible, key=lambda r: r["macro_f1"]) if eligible else None
        if best is None:
            selected[model_name] = None
            continue
        selected[model_name] = {"configuration": best["configuration"], "validation_macro_f1": best["macro_f1"]}

    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report = {"feature_schema": manifest["schema"], "feature_manifest_sha256": sha256(feature_dir / "manifest.json"),
              "split_manifest_sha256": sha256(split_dir / "split_manifest.csv"), "test_scored": False,
              "logistic_solver": {"optimizer": "PyTorch LBFGS", "max_iter": LR_MAX_ITER,
                                  "max_eval": LR_MAX_EVAL,
                                  "tolerance_grad": LR_TOLERANCE_GRAD,
                                  "tolerance_change": LR_TOLERANCE_CHANGE,
                                  "accepts_gradient_or_change_tolerance_stop": True},
              "fit_partition": "train", "selection_partition": "validation",
              "train_samples": len(train_ids), "validation_samples": len(val_ids),
              "majority_label_train": majority_label, "selected_by_validation": selected, "results": results}
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "error_type", "majority", "best_lr", "best_tree"])
        writer.writeheader()
        lr_cfg = selected["LogisticRegression"]["configuration"] if selected["LogisticRegression"] else None
        tree_cfg = selected["DecisionTree"]["configuration"] if selected["DecisionTree"] else None
        best_lr_key = (("LogisticRegression", lr_cfg["C"], lr_cfg["class_weight"] == "balanced")
                       if lr_cfg else None)
        best_tree_key = (("DecisionTree", tree_cfg["max_depth"], tree_cfg["min_samples_leaf"],
                          tree_cfg["class_weight"] == "balanced") if tree_cfg else None)
        for i, sid in enumerate(val_ids):
            writer.writerow({"sample_id": sid, "error_type": y_val[i], "majority": majority_label,
                             "best_lr": validation_predictions[best_lr_key][i] if best_lr_key else "",
                             "best_tree": validation_predictions[best_tree_key][i] if best_tree_key else ""})
    finished_at = datetime.now(timezone.utc)
    fit_seconds_by_model = {
        model_name: sum(row.get("fit_seconds", 0.0) for row in results if row["model"] == model_name)
        for model_name in ("LogisticRegression", "DecisionTree")
    }
    report["timing"] = {
        "started_at_utc": run_started_at.isoformat(timespec="milliseconds"),
        "finished_at_utc": finished_at.isoformat(timespec="milliseconds"),
        "wall_seconds": perf_counter() - run_clock,
        "data_loading_seconds": data_loading_seconds,
        "model_fit_seconds_by_model": fit_seconds_by_model,
        "model_fit_seconds_total": sum(fit_seconds_by_model.values()),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.split, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
