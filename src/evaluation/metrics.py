"""Metrics for the NUCLE five-label classification experiments."""
from __future__ import annotations

from typing import Any

from src.data.constants import TARGET_LABELS

LABELS = tuple(TARGET_LABELS)


def scores(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Return fixed-label macro metrics and a true-row/predicted-column matrix."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must be non-empty and aligned")
    index = {label: i for i, label in enumerate(LABELS)}
    if any(label not in index for label in (*y_true, *y_pred)):
        raise ValueError("Metric input contains an unknown label")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[index[truth]][index[prediction]] += 1
    per_label = {}
    f1_values = []
    for i, label in enumerate(LABELS):
        true_positive = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    return {
        "macro_f1": sum(f1_values) / len(LABELS),
        "accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / len(y_true),
        "per_label": per_label,
        "confusion_matrix_true_rows_predicted_columns": matrix,
    }


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Return accuracy for a non-empty, aligned pair of label sequences."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Accuracy inputs must be non-empty and aligned")
    return sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)
