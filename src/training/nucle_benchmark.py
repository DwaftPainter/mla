"""Train sparse Logistic Regression and Decision Tree on S/L/P/M/D features.

Run from the repository root after feature extraction:
    .venv/Scripts/python.exe src/training/nucle_benchmark.py

Only train labels fit parameters; validation Macro-F1 selects a configuration.
The test partition is excluded from fitting, selection, and reported metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import DATA_ROOT, PROJECT_ROOT, TARGET_LABELS

FEATURE_DIR = DATA_ROOT / "features" / "nucle_features_v1"
SPLIT_DIR = DATA_ROOT / "split" / "nucle_group_split"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "nucle_feature_benchmark_solver_v2"
LABELS = tuple(TARGET_LABELS)
SEED = 20261013
LR_MAX_ITER = 1000
LR_MAX_EVAL = 1250
LR_TOLERANCE_GRAD = 1e-6
LR_TOLERANCE_CHANGE = 1e-9


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


def scores(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    index = {label: i for i, label in enumerate(LABELS)}
    cm = [[0] * len(LABELS) for _ in LABELS]
    for a, b in zip(y_true, y_pred, strict=True):
        cm[index[a]][index[b]] += 1
    per = {}
    f1s = []
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[label] = {"precision": p, "recall": r, "f1": f, "support": support}
        f1s.append(f)
    return {"macro_f1": float(np.mean(f1s)), "accuracy": sum(cm[i][i] for i in range(len(LABELS))) / len(y_true),
            "per_label": per, "confusion_matrix_true_rows_predicted_columns": cm}


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    require(len(y_true) == len(y_pred) and bool(y_true), "Accuracy inputs must be non-empty and aligned")
    return sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def balanced_weights(y: list[int]) -> torch.Tensor:
    counts = Counter(y)
    return torch.tensor([len(y) / (len(LABELS) * counts.get(i, 1)) for i in range(len(LABELS))], dtype=torch.float32)


def fit_logistic(x: torch.Tensor, y: list[int], c_value: float, balanced: bool,
                 max_iter: int = LR_MAX_ITER):
    torch.manual_seed(SEED)
    n, d = x.shape
    labels = torch.tensor(y, dtype=torch.long)
    weights = balanced_weights(y) if balanced else None
    w = torch.nn.Parameter(torch.zeros(d, len(LABELS), dtype=torch.float32))
    b = torch.nn.Parameter(torch.zeros(len(LABELS), dtype=torch.float32))
    optimizer = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter,
                                  max_eval=LR_MAX_EVAL,
                                  tolerance_grad=LR_TOLERANCE_GRAD,
                                  tolerance_change=LR_TOLERANCE_CHANGE,
                                  line_search_fn="strong_wolfe")
    eval_count = 0
    def closure():
        nonlocal eval_count
        optimizer.zero_grad()
        logits = torch.sparse.mm(x, w) + b
        losses = F.cross_entropy(logits, labels, weight=weights, reduction="mean")
        objective = losses + 0.5 * w.square().sum() / (c_value * n)
        objective.backward()
        eval_count += 1
        return objective
    optimizer.step(closure)
    optimizer_iters = int(optimizer.state[w].get("n_iter", 0))
    optimizer.zero_grad()
    final_objective = F.cross_entropy(torch.sparse.mm(x, w) + b, labels, weight=weights)
    final_objective = final_objective + 0.5 * w.square().sum() / (c_value * n)
    final_objective.backward()
    grad_inf_norm = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
    final_loss = float(final_objective.detach())
    optimizer.zero_grad()
    gradient_stop = grad_inf_norm <= LR_TOLERANCE_GRAD
    change_stop = optimizer_iters < max_iter and eval_count < LR_MAX_EVAL
    converged = gradient_stop or change_stop
    return w.detach(), b.detach(), {"iterations": optimizer_iters, "iterations_max": max_iter,
                                    "closure_evaluations": eval_count, "max_evaluations": LR_MAX_EVAL,
                                    "final_objective": final_loss,
                                    "gradient_inf_norm": grad_inf_norm,
                                    "tolerance_grad": LR_TOLERANCE_GRAD,
                                    "tolerance_change": LR_TOLERANCE_CHANGE,
                                    "termination": "gradient_tolerance" if gradient_stop else
                                                  "change_tolerance" if change_stop else
                                                  "iteration_limit" if optimizer_iters >= max_iter else "evaluation_limit",
                                    "converged": converged}


class GiniTree:
    """Small deterministic CART for sparse one-hot and numeric feature rows."""
    def __init__(self, max_depth: int, min_samples_leaf: int, balanced: bool, numeric_features: set[int]):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.balanced, self.numeric_features = balanced, numeric_features

    def fit(self, rows: list[dict[int, float]], y: list[int]):
        self.rows, self.y = rows, y
        counts = Counter(y)
        self.sample_weights = [len(y) / (len(LABELS) * counts[v]) if self.balanced else 1.0 for v in y]
        self.root = self._build(list(range(len(y))), 0)
        return self

    def _impurity(self, ids: list[int]) -> float:
        total = sum(self.sample_weights[i] for i in ids)
        if not total: return 0.0
        counts = [0.0] * len(LABELS)
        for i in ids: counts[self.y[i]] += self.sample_weights[i]
        return 1.0 - sum((v / total) ** 2 for v in counts)

    def _build(self, ids: list[int], depth: int):
        class_counts = Counter(self.y[i] for i in ids)
        majority = min(range(len(LABELS)), key=lambda i: (-class_counts[i], i))
        node = {"prediction": majority}
        if depth >= self.max_depth or len(class_counts) == 1 or len(ids) < 2 * self.min_samples_leaf:
            return node
        parent_impurity = self._impurity(ids)
        postings: dict[int, list[int]] = {}
        for i in ids:
            for feature, value in self.rows[i].items():
                if value:
                    postings.setdefault(feature, []).append(i)
        best = None
        node_weight = sum(self.sample_weights[i] for i in ids)
        total_label_weights = [sum(self.sample_weights[i] for i in ids if self.y[i] == label)
                               for label in range(len(LABELS))]
        def gini(counts, total):
            return 1.0 - sum((n / total) ** 2 for n in counts if n) if total else 0.0
        for feature in sorted(postings):
            if feature in self.numeric_features:
                ordered = sorted((self.rows[i].get(feature, 0.0), i) for i in ids)
                left_counts = [0.0] * len(LABELS)
                left_raw = 0
                for pos in range(len(ordered) - 1):
                    value, sample_id = ordered[pos]
                    left_counts[self.y[sample_id]] += self.sample_weights[sample_id]
                    left_raw += 1
                    next_value = ordered[pos + 1][0]
                    if value == next_value or left_raw < self.min_samples_leaf or len(ids) - left_raw < self.min_samples_leaf:
                        continue
                    right_counts = [total_label_weights[label] - left_counts[label] for label in range(len(LABELS))]
                    wl, wr = sum(left_counts), sum(right_counts)
                    gain = parent_impurity - wl / node_weight * gini(left_counts, wl) - wr / node_weight * gini(right_counts, wr)
                    threshold = (value + next_value) / 2
                    candidate = (gain, -feature, -threshold, feature, threshold)
                    if gain > 1e-12 and (best is None or candidate[:3] > best[:3]): best = candidate
            else:
                present = postings[feature]
                if len(present) < self.min_samples_leaf or len(ids) - len(present) < self.min_samples_leaf: continue
                left_set = set(present)
                left_counts = [0.0] * len(LABELS)
                for i in present: left_counts[self.y[i]] += self.sample_weights[i]
                right_counts = [total_label_weights[label] - left_counts[label] for label in range(len(LABELS))]
                wl, wr = sum(left_counts), sum(right_counts)
                gain = parent_impurity - wl / node_weight * gini(left_counts, wl) - wr / node_weight * gini(right_counts, wr)
                candidate = (gain, -feature, -0.5, feature, 0.5)
                if gain > 1e-12 and (best is None or candidate[:3] > best[:3]): best = candidate
        if best is None: return node
        _, _, _, feature, threshold = best
        left = [i for i in ids if self.rows[i].get(feature, 0.0) <= threshold]
        right = [i for i in ids if self.rows[i].get(feature, 0.0) > threshold]
        node.update(feature=feature, threshold=threshold,
                    left=self._build(left, depth + 1), right=self._build(right, depth + 1))
        return node

    def predict_one(self, row: dict[int, float]) -> int:
        node = self.root
        while "feature" in node:
            node = node["left"] if row.get(node["feature"], 0.0) <= node["threshold"] else node["right"]
        return node["prediction"]

    def predict(self, rows: list[dict[int, float]]) -> list[int]:
        return [self.predict_one(row) for row in rows]


def run(feature_dir: Path = FEATURE_DIR, split_dir: Path = SPLIT_DIR, output_dir: Path = DEFAULT_OUTPUT) -> dict:
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
    results, validation_predictions = [], {}
    y_train_labels = [LABELS[i] for i in y_train]
    majority_label = min(LABELS, key=lambda label: (-y_train_labels.count(label), LABELS.index(label)))
    results.append({"model": "Majority", "configuration": {"label": majority_label},
                    "train_accuracy": accuracy(y_train_labels, [majority_label] * len(y_train_labels)),
                    **scores(y_val, [majority_label] * len(y_val))})

    for c_value in (0.01, 0.1, 1.0, 10.0):
        for balanced in (False, True):
            w, b, convergence = fit_logistic(x_train, y_train, c_value, balanced)
            with torch.no_grad():
                train_idx = (torch.sparse.mm(x_train, w) + b).argmax(dim=1).tolist()
                val_idx = (torch.sparse.mm(x_val, w) + b).argmax(dim=1).tolist()
            pred = [LABELS[i] for i in val_idx]
            train_pred = [LABELS[i] for i in train_idx]
            key = ("LogisticRegression", c_value, balanced)
            validation_predictions[key] = pred
            results.append({"model": "LogisticRegression", "configuration": {"C": c_value,
                            "class_weight": "balanced" if balanced else None},
                           "convergence": convergence, "train_accuracy": accuracy(y_train_labels, train_pred),
                           **scores(y_val, pred)})

    for depth in (4, 8, 12):
        for leaf in (5, 20):
            for balanced in (False, True):
                tree = GiniTree(depth, leaf, balanced, numeric_features).fit(sparse_train, y_train)
                pred = [LABELS[i] for i in tree.predict(sparse_val)]
                train_pred = [LABELS[i] for i in tree.predict(sparse_train)]
                validation_predictions[("DecisionTree", depth, leaf, balanced)] = pred
                results.append({"model": "DecisionTree", "configuration": {"max_depth": depth,
                                "min_samples_leaf": leaf, "class_weight": "balanced" if balanced else None},
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
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_DIR)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.split, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
