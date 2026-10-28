"""Synthetic integration tests; no corpus, download or research training."""
import csv
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", reason="Install requirements-bert.txt for BERT tests")
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.training import bert_small_experiment as experiment


@pytest.fixture
def setup_experiment(tmp_path):
    source = tmp_path / "pretrained"
    source.mkdir()
    vocab = source / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                "they", "goes", "home", ".", "left", "right"]) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tokenizer.save_pretrained(source)
    torch.manual_seed(7)
    base = BertForSequenceClassification(BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, num_labels=5, max_position_embeddings=32,
        id2label=dict(enumerate(experiment.LABELS)),
        label2id={label: i for i, label in enumerate(experiment.LABELS)},
    ))
    base.save_pretrained(source)
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps({"model": "synthetic-bert", "path": str(source)}), encoding="utf-8")
    samples = tmp_path / "samples.jsonl"
    split = tmp_path / "split.csv"
    rows, splits = [], []
    for part, repeats in (("train", 2), ("validation", 1), ("test", 1)):
        for index, label in enumerate(experiment.LABELS):
            for repeat in range(repeats):
                sid = f"{part}-{index}-{repeat}"
                rows.append({"sample_id": sid, "sentence": "they goes home .", "error_start": 5,
                             "error_end": 9, "error_span": "goes", "error_type": label})
                splits.append({"sample_id": sid, "partition": part, "group_id": f"{part}-group"})
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with split.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(splits[0]))
        writer.writeheader()
        writer.writerows(splits)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "samples": str(samples), "split": str(split), "model_source": str(source_file),
        "expected_counts": {"train": 10, "validation": 5, "test": 5},
        "readouts": ["span_mean"], "encoder_learning_rates": [1e-3],
        "head_learning_rates": [1e-3], "seeds": [11], "max_epochs": 2,
        "patience": 2, "batch_size": 5, "max_length": 16, "threads": 1,
        "dropout": .1, "device": "cpu", "warmup_ratio": 0.,
    }), encoding="utf-8")
    return experiment.load_config(config), tmp_path


def test_complete_lifecycle_and_lock_protection(setup_experiment):
    config, root = setup_experiment
    output = root / "experiment"
    experiment.train(config, output)
    run = output / experiment.run_name(next(experiment.variants(config)), 11)
    history = experiment.read_json(run / "history.json")
    assert len(history) == 2
    assert history[-1]["train"]["support"] == 10
    assert history[-1]["validation"]["support"] == 5
    assert (run / "history.csv").exists()
    assert (run / "learning_curves.png").exists()
    assert "test" not in experiment.read_json(output / "data_audit.json")
    assert not (output / "selection.json").exists()
    experiment.select(output)
    lock = experiment.read_json(output / "selection.json")
    assert lock["primary_seed"] == 11
    with pytest.raises(ValueError, match="already locked"):
        experiment.train(config, output, resume=True)
    test_output = root / "test-output"
    experiment.test(output / "selection.json", test_output)
    summary = experiment.read_json(test_output / "summary.json")
    assert summary["test_scored"] is True
    assert summary["runs"][0]["metrics"]["support"] == 5
    assert summary["test_macro_f1"]["sample_std"] is None
    assert (test_output / "per_class_metrics.csv").exists()
    with pytest.raises(FileExistsError):
        experiment.test(output / "selection.json", test_output)
    (run / "best" / "readout.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint changed"):
        experiment.test(output / "selection.json", root / "tampered-test")
    assert not (root / "tampered-test").exists()


def test_resume_restores_optimizer_scheduler_and_dropout_rng(setup_experiment, monkeypatch):
    config, root = setup_experiment
    uninterrupted = root / "uninterrupted"
    experiment.train(config, uninterrupted)
    interrupted = root / "interrupted"
    original = experiment.evaluate
    calls = 0

    def fail_in_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "evaluate", fail_in_second_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.train(config, interrupted)
    monkeypatch.setattr(experiment, "evaluate", original)
    experiment.train(config, interrupted, resume=True)
    name = experiment.run_name(next(experiment.variants(config)), 11)
    expected = torch.load(uninterrupted / name / "last.pt", map_location="cpu", weights_only=False)
    actual = torch.load(interrupted / name / "last.pt", map_location="cpu", weights_only=False)
    for key, tensor in expected["model"].items():
        assert torch.equal(tensor, actual["model"][key]), key
    assert expected["history"][-1]["train"] == actual["history"][-1]["train"]
    assert expected["history"][-1]["validation"] == actual["history"][-1]["validation"]


def test_early_stopping_keeps_small_improvement_as_best(setup_experiment, monkeypatch):
    config, root = setup_experiment
    config.update(max_epochs=6, patience=2, min_delta=.0001)
    original = experiment.evaluate
    validation_calls = 0

    def controlled_validation(model, batches, device):
        nonlocal validation_calls
        metrics, predictions = original(model, batches, device)
        if len(batches.dataset) == 5:
            metrics["macro_f1"] = [.2, .20005, .19, .20005][validation_calls]
            validation_calls += 1
        return metrics, predictions

    monkeypatch.setattr(experiment, "evaluate", controlled_validation)
    output = root / "early-stop"
    experiment.train(config, output)
    run = output / experiment.run_name(next(experiment.variants(config)), 11)
    summary = experiment.read_json(run / "summary.json")
    assert summary["epochs_completed"] == 3
    assert summary["stop_reason"] == "early_stopping"
    assert summary["best_epoch"] == 2
    assert summary["best"]["validation"]["macro_f1"] == .20005


def test_train_validation_loader_does_not_tokenize_test(setup_experiment):
    config, _ = setup_experiment
    path = Path(config["samples"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["sample_id"].startswith("test-"):
            row["error_span"] = "invalid test span should never be tokenized during training"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _, tokenizer = experiment.source_and_tokenizer(config)
    data, _ = experiment.load_rows(config, tokenizer, ("train", "validation"))
    assert set(data) == {"train", "validation"}
    with pytest.raises(ValueError, match="Invalid original span"):
        experiment.load_rows(config, tokenizer, ("test",))


def test_training_weights_only_use_train_counts():
    rows = [(str(i), {}, label) for i, label in enumerate([0, 0, 0, 0, 1, 2, 3, 4])]
    weights = experiment.train_weights(rows, "balanced", torch.device("cpu"))
    assert weights[0].item() == pytest.approx(.4)
    assert weights[1:].tolist() == pytest.approx([1.6] * 4)


def test_selection_uses_seed_mean_not_best_single_run(setup_experiment, monkeypatch):
    config, root = setup_experiment
    config["seeds"] = [11, 12]
    config["readouts"] = ["cls", "span_mean"]
    output = root / "selection-only"
    output.mkdir()
    experiment.write_json(output / "manifest.json", {"config": config, "code": experiment.code_hashes()})
    values = {("cls", 11): .9, ("cls", 12): .1, ("span_mean", 11): .6, ("span_mean", 12): .6}

    def fake_validate(directory, variant, seed):
        return {"best": {"validation": {"macro_f1": values[variant["readout"], seed]}}}

    monkeypatch.setattr(experiment, "validate_run", fake_validate)
    experiment.select(output)
    lock = experiment.read_json(output / "selection.json")
    assert lock["selected"]["variant"]["readout"] == "span_mean"
    assert lock["primary_seed"] == 11
    assert lock["selected"]["validation_macro_f1"]["mean"] == .6
