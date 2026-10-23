import csv
import json
from pathlib import Path
import shutil
import tempfile

from src.training.nucle_benchmark import GiniTree, LABELS, accuracy, load_rows, run, scores


def test_sparse_gini_tree_learns_a_binary_split():
    rows = [{0: 1.0}, {1: 1.0}, {0: 1.0}, {1: 1.0}]
    labels = [LABELS.index("Prep"), LABELS.index("Nn"), LABELS.index("Prep"), LABELS.index("Nn")]
    tree = GiniTree(max_depth=2, min_samples_leaf=1, balanced=False, numeric_features=set()).fit(rows, labels)
    assert tree.predict(rows) == labels


def test_validation_scores_use_fixed_five_class_macro_average():
    result = scores(["Prep", "Prep"], ["Prep", "Nn"])
    assert len(result["per_label"]) == 5
    assert result["per_label"]["Prep"]["support"] == 2
    assert len(result["confusion_matrix_true_rows_predicted_columns"]) == 5


def test_accuracy_requires_aligned_nonempty_predictions():
    assert accuracy(["Prep", "Nn"], ["Prep", "Prep"]) == 0.5


def test_loader_does_not_retain_or_validate_test_features_and_labels():
    root = Path(tempfile.mkdtemp(prefix="nucle-benchmark-test-", dir=Path.cwd()))
    try:
        feature_dir, split_dir = root / "features", root / "split"
        feature_dir.mkdir(); split_dir.mkdir()
        with (split_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "partition"])
            writer.writeheader()
            writer.writerows([{"sample_id": "tr", "partition": "train"},
                              {"sample_id": "va", "partition": "validation"},
                              {"sample_id": "te", "partition": "test"}])
        (feature_dir / "features.jsonl").write_text("\n".join(json.dumps(row) for row in [
            {"sample_id": "tr", "partition": "train", "x": {"0": 1}},
            {"sample_id": "va", "partition": "validation", "x": {"0": 0}},
            {"sample_id": "te", "partition": "test", "x": "must not be converted"},
        ]), encoding="utf-8")
        (feature_dir / "targets.jsonl").write_text("\n".join(json.dumps(row) for row in [
            {"sample_id": "tr", "partition": "train", "error_type": "Prep"},
            {"sample_id": "va", "partition": "validation", "error_type": "Prep"},
            {"sample_id": "te", "partition": "test", "error_type": "NOT_A_LABEL"},
        ]), encoding="utf-8")
        x, y, _, train_ids, val_ids = load_rows(feature_dir, split_dir)
        assert train_ids == ["tr"] and val_ids == ["va"]
        assert "te" not in x and "te" not in y
    finally:
        shutil.rmtree(root)


def test_benchmark_report_records_run_and_per_model_timing():
    root = Path(tempfile.mkdtemp(prefix="nucle-benchmark-timing-", dir=Path.cwd()))
    try:
        feature_dir, split_dir, output_dir = root / "features", root / "split", root / "report"
        feature_dir.mkdir(); split_dir.mkdir()
        (feature_dir / "manifest.json").write_text(json.dumps({"schema": "nucle-slpmd-v1"}), encoding="utf-8")
        (feature_dir / "encoder.json").write_text(json.dumps({
            "schema": "nucle-slpmd-v1", "fit_partition": "train",
            "feature_names": [f"C|label_{i}" for i in range(len(LABELS))],
        }), encoding="utf-8")

        split_rows, feature_rows, target_rows = [], [], []
        for i, label in enumerate(LABELS):
            for repeat in range(3):
                sid = f"train_{i}_{repeat}"
                split_rows.append({"sample_id": sid, "partition": "train"})
                feature_rows.append({"sample_id": sid, "partition": "train", "x": {str(i): 1.0}})
                target_rows.append({"sample_id": sid, "partition": "train", "error_type": label})
            for partition in ("validation", "test"):
                sid = f"{partition}_{i}"
                split_rows.append({"sample_id": sid, "partition": partition})
                # Test payload must be ignored by the runner.
                x = {str(i): 1.0} if partition == "validation" else "not loaded"
                y = label if partition == "validation" else "not read"
                feature_rows.append({"sample_id": sid, "partition": partition, "x": x})
                target_rows.append({"sample_id": sid, "partition": partition, "error_type": y})

        with (split_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "partition"])
            writer.writeheader(); writer.writerows(split_rows)
        (feature_dir / "features.jsonl").write_text(
            "\n".join(json.dumps(row) for row in feature_rows), encoding="utf-8"
        )
        (feature_dir / "targets.jsonl").write_text(
            "\n".join(json.dumps(row) for row in target_rows), encoding="utf-8"
        )

        report = run(feature_dir, split_dir, output_dir)
        timing = report["timing"]
        assert timing["started_at_utc"].endswith("+00:00")
        assert timing["finished_at_utc"].endswith("+00:00")
        assert timing["wall_seconds"] >= timing["model_fit_seconds_total"] >= 0
        assert set(timing["model_fit_seconds_by_model"]) == {"LogisticRegression", "DecisionTree"}
        assert all(row["fit_seconds"] >= 0 for row in report["results"] if "fit_seconds" in row)
        saved = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert saved["timing"] == timing
    finally:
        shutil.rmtree(root)
