"""Deterministic Gini CART for sparse NUCLE feature dictionaries."""
from __future__ import annotations

from collections import Counter

from src.data.constants import TARGET_LABELS

LABELS = tuple(TARGET_LABELS)


class GiniTree:
    """Small CART implementation for one-hot and numeric sparse features."""

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        balanced: bool,
        numeric_features: set[int],
    ):
        if max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("max_depth and min_samples_leaf must be positive")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.balanced = balanced
        self.numeric_features = numeric_features

    def fit(self, rows: list[dict[int, float]], y: list[int]) -> "GiniTree":
        if not rows or len(rows) != len(y):
            raise ValueError("Feature rows and non-empty targets must be aligned")
        if any(label < 0 or label >= len(LABELS) for label in y):
            raise ValueError("Target index outside the fixed NUCLE label set")
        self.rows, self.y = rows, y
        counts = Counter(y)
        self.sample_weights = [
            len(y) / (len(LABELS) * counts[label]) if self.balanced else 1.0
            for label in y
        ]
        self.root = self._build(list(range(len(y))), 0)
        return self

    def _impurity(self, ids: list[int]) -> float:
        total = sum(self.sample_weights[i] for i in ids)
        if not total:
            return 0.0
        counts = [0.0] * len(LABELS)
        for i in ids:
            counts[self.y[i]] += self.sample_weights[i]
        return 1.0 - sum((value / total) ** 2 for value in counts)

    def _build(self, ids: list[int], depth: int) -> dict:
        class_counts = Counter(self.y[i] for i in ids)
        majority = min(range(len(LABELS)), key=lambda i: (-class_counts[i], i))
        node = {"prediction": majority}
        if (
            depth >= self.max_depth
            or len(class_counts) == 1
            or len(ids) < 2 * self.min_samples_leaf
        ):
            return node

        parent_impurity = self._impurity(ids)
        postings: dict[int, list[int]] = {}
        for i in ids:
            for feature, value in self.rows[i].items():
                if value:
                    postings.setdefault(feature, []).append(i)

        best = None
        node_weight = sum(self.sample_weights[i] for i in ids)
        total_label_weights = [
            sum(self.sample_weights[i] for i in ids if self.y[i] == label)
            for label in range(len(LABELS))
        ]

        def gini(counts: list[float], total: float) -> float:
            return 1.0 - sum((value / total) ** 2 for value in counts if value) if total else 0.0

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
                    if (
                        value == next_value
                        or left_raw < self.min_samples_leaf
                        or len(ids) - left_raw < self.min_samples_leaf
                    ):
                        continue
                    right_counts = [
                        total_label_weights[label] - left_counts[label]
                        for label in range(len(LABELS))
                    ]
                    wl, wr = sum(left_counts), sum(right_counts)
                    gain = (
                        parent_impurity
                        - wl / node_weight * gini(left_counts, wl)
                        - wr / node_weight * gini(right_counts, wr)
                    )
                    threshold = (value + next_value) / 2
                    candidate = (gain, -feature, -threshold, feature, threshold)
                    if gain > 1e-12 and (best is None or candidate[:3] > best[:3]):
                        best = candidate
            else:
                present = postings[feature]
                if (
                    len(present) < self.min_samples_leaf
                    or len(ids) - len(present) < self.min_samples_leaf
                ):
                    continue
                left_counts = [0.0] * len(LABELS)
                for i in present:
                    left_counts[self.y[i]] += self.sample_weights[i]
                right_counts = [
                    total_label_weights[label] - left_counts[label]
                    for label in range(len(LABELS))
                ]
                wl, wr = sum(left_counts), sum(right_counts)
                gain = (
                    parent_impurity
                    - wl / node_weight * gini(left_counts, wl)
                    - wr / node_weight * gini(right_counts, wr)
                )
                candidate = (gain, -feature, -0.5, feature, 0.5)
                if gain > 1e-12 and (best is None or candidate[:3] > best[:3]):
                    best = candidate

        if best is None:
            return node
        _, _, _, feature, threshold = best
        left = [i for i in ids if self.rows[i].get(feature, 0.0) <= threshold]
        right = [i for i in ids if self.rows[i].get(feature, 0.0) > threshold]
        node.update(
            feature=feature,
            threshold=threshold,
            left=self._build(left, depth + 1),
            right=self._build(right, depth + 1),
        )
        return node

    def predict_one(self, row: dict[int, float]) -> int:
        if not hasattr(self, "root"):
            raise RuntimeError("Call fit before predict")
        node = self.root
        while "feature" in node:
            node = (
                node["left"]
                if row.get(node["feature"], 0.0) <= node["threshold"]
                else node["right"]
            )
        return node["prediction"]

    def predict(self, rows: list[dict[int, float]]) -> list[int]:
        return [self.predict_one(row) for row in rows]
