import random

import numpy as np
import pytest

from src.evaluation.bert_final_analysis import LABELS, balanced_take, choose_audit, group_bootstrap
from src.evaluation.metrics import scores


def test_group_bootstrap_matches_explicit_group_expansion():
    metadata = [{"group_id": "a" if i < 5 else "b", "error_type": LABELS[i % 5]} for i in range(10)]
    pred = [row["error_type"] for row in metadata]
    pred[1] = "SVA"
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    distributions = group_bootstrap(metadata, {"model": pred, "same": pred}, ["a", "b"], weights)
    for j, w in enumerate(weights):
        ids = [i for group, n in zip(["a", "b"], w) for _ in range(int(n))
               for i, row in enumerate(metadata) if row["group_id"] == group]
        expected = scores([metadata[i]["error_type"] for i in ids], [pred[i] for i in ids])["macro_f1"]
        assert distributions["model"][j] == pytest.approx(expected)
    assert np.array_equal(distributions["model"], distributions["same"])


def test_bootstrap_rejects_unmatched_groups():
    with pytest.raises(ValueError, match="Group set"):
        group_bootstrap([{"group_id": "a", "error_type": "SVA"}], {"model": ["SVA"]}, ["b"], np.array([[1]]))


def test_audit_selection_is_unique_deterministic_and_directional():
    rows = []
    for kind in range(4):
        for i in range(50):
            gold = LABELS[i % 5]
            wrong = LABELS[(i + 1) % 5]
            rows.append({"sample_id": f"{kind}-{i:03}", "error_type": gold,
                         "BERT": gold if kind in (1, 3) else wrong,
                         "LR_03": gold if kind in (0, 3) else wrong})
    a = choose_audit(rows)
    b = choose_audit(list(reversed(rows)))
    assert a == b
    assert len(a) == len({r["sample_id"] for _, r in a}) == 40
    assert sum(category == "BERT_wrong_LR_correct" and row["error_type"] == "SVA" for category, row in a) >= 6
    assert all(row["BERT"] != row["error_type"] == row["LR_03"] for category, row in a if category == "BERT_wrong_LR_correct")


def test_audit_rejects_insufficient_candidates():
    with pytest.raises(ValueError, match="Insufficient"):
        balanced_take([], 1, random.Random(1))
