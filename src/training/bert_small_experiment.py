"""BERT-small train/validation search, selection lock and separate test scoring.

Run with ``python -m src.training.bert_small_experiment --help``.
Historical BERT outputs and their protocols are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import LABELS, scores
from src.models.bert_span_readout import BertReadout, target_mask
from src.training.bert_span_classifier import encode_span

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/bert_small_training.json"
CODE_FILES = (
    "src/training/bert_small_experiment.py", "src/training/bert_span_classifier.py",
    "src/models/bert_span_readout.py", "src/evaluation/metrics.py", "src/data/constants.py",
)
DEFAULTS = {
    "expected_counts": {"train": 11052, "validation": 2368, "test": 2368},
    "readouts": ["cls", "span_mean"], "encoder_learning_rates": [5e-5],
    "head_learning_rates": [1e-4, 1e-3], "seeds": [20261013, 20261014, 20261015],
    "max_epochs": 20, "patience": 4, "min_delta": 1e-4, "batch_size": 32,
    "max_length": 128, "dropout": .1, "weight_decay": .01, "warmup_ratio": .1,
    "gradient_clip": 1., "class_weight": "none", "device": "cpu", "threads": 4,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): digest(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def code_hashes():
    return {name: digest(ROOT / name) for name in CODE_FILES}


def load_config(path):
    supplied = read_json(path)
    unknown = set(supplied) - set(DEFAULTS) - {"samples", "split", "model_source"}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    for name in ("samples", "split", "model_source"):
        config[name] = str(resolve(config[name]))
    for name in ("max_epochs", "patience", "batch_size", "threads", "max_length"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["max_length"] < 5:
        raise ValueError("max_length must leave room for span markers and content")
    for name in ("readouts", "encoder_learning_rates", "head_learning_rates", "seeds"):
        values = config[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be a non-empty list without duplicates")
    if set(config["readouts"]) - {"cls", "span_mean"}:
        raise ValueError("readouts must contain cls and/or span_mean")
    if any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be non-negative integers")
    for value in config["encoder_learning_rates"] + config["head_learning_rates"]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates must be finite and positive")
    for name in ("min_delta", "weight_decay", "gradient_clip", "dropout", "warmup_ratio"):
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name}")
    if config["gradient_clip"] <= 0 or config["dropout"] >= 1 or config["warmup_ratio"] >= 1:
        raise ValueError("Invalid clipping/dropout/warmup")
    if config["class_weight"] not in ("none", "balanced"):
        raise ValueError("class_weight must be none or balanced")
    if config["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    counts = config["expected_counts"]
    if set(counts) != {"train", "validation", "test"} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("expected_counts must provide positive train/validation/test counts")
    return config


def runtime(config):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment; choose cpu or install a CUDA PyTorch build")
    return torch.device(device)


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__,
            "device": str(device), "threads": torch.get_num_threads(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def source_and_tokenizer(config):
    source = read_json(config["model_source"])
    source["path"] = str(resolve(source["path"]))
    tokenizer = BertTokenizerFast.from_pretrained(source["path"], local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SPAN_START]", "[SPAN_END]"]})
    return source, tokenizer


def input_identity(config, source):
    return {
        "samples": digest(config["samples"]), "split": digest(config["split"]),
        "model_source": digest(config["model_source"]),
        "pretrained_files": tree_hashes(Path(source["path"])),
    }


def load_rows(config, tokenizer, partitions):
    """Only tokenize/retain requested partitions; no test loader during search."""
    with Path(config["split"]).open(encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    split = {row["sample_id"]: row["partition"] for row in split_rows}
    if len(split) != len(split_rows) or set(split.values()) != {"train", "validation", "test"}:
        raise ValueError("Duplicate IDs or invalid partition names in split")
    # Check available group/essay metadata without using test text or labels.
    for key in ("group_id", "essay_id", "essay_index"):
        seen = {}
        for row in split_rows:
            if row.get(key):
                previous = seen.setdefault(row[key], row["partition"])
                if previous != row["partition"]:
                    raise ValueError(f"{key} crosses partitions")
    counts = Counter(split.values())
    if dict(counts) != config["expected_counts"]:
        raise ValueError(f"Unexpected split sizes: {dict(counts)}")
    datasets = {part: [] for part in partitions}
    stats = {part: {"count": 0, "truncated": 0, "labels": {}} for part in partitions}
    seen = set()
    with Path(config["samples"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in split or sid in seen:
                raise ValueError(f"Missing split or duplicate sample ID: {sid}")
            seen.add(sid)
            part = split[sid]
            if part not in datasets:
                continue
            features, stat = encode_span({key: row[key] for key in (
                "sentence", "error_start", "error_end", "error_span")}, tokenizer, config["max_length"])
            marker_ids = tokenizer.convert_tokens_to_ids(["[SPAN_START]", "[SPAN_END]"])
            if any(features["input_ids"].count(i) != 1 for i in marker_ids):
                raise ValueError(f"Invalid or embedded span markers: {sid}")
            datasets[part].append((sid, features, LABELS.index(row["error_type"])))
            stats[part]["count"] += 1
            stats[part]["truncated"] += int(stat["truncated"])
    if seen != set(split):
        raise ValueError("Prepared samples and split IDs differ")
    for part, rows in datasets.items():
        rows.sort(key=lambda row: row[0])
        stats[part]["labels"] = dict(Counter(LABELS[row[2]] for row in rows))
        if len(rows) != counts[part] or set(stats[part]["labels"]) != set(LABELS):
            raise ValueError(f"Missing samples or classes in {part}")
    return datasets, stats


def loader(rows, tokenizer, batch_size, seed=None):
    def collate(batch):
        ids, features, labels = zip(*batch)
        encoded = tokenizer.pad(list(features), padding=True, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return ids, encoded
    return DataLoader(rows, batch_size=batch_size, shuffle=seed is not None, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed or 0), num_workers=0)


def make_model(source, tokenizer, config, readout, device):
    base = BertForSequenceClassification.from_pretrained(
        source["path"], num_labels=len(LABELS), local_files_only=True,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        hidden_dropout_prob=config["dropout"], attention_probs_dropout_prob=config["dropout"],
    )
    if config["max_length"] > base.config.max_position_embeddings:
        raise ValueError("max_length exceeds pretrained position embeddings")
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    start, end = tokenizer.convert_tokens_to_ids(["[SPAN_START]", "[SPAN_END]"])
    return BertReadout(base, readout, start, end).to(device)


def optimizer_for(model, config, variant):
    groups = {}
    for name, parameter in model.named_parameters():
        # Pooler and classifier form the head; embeddings and transformer use encoder LR.
        head = name.startswith(("base.classifier.", "base.bert.pooler."))
        lr = variant["head_lr"] if head else variant["encoder_lr"]
        decay = 0. if name.endswith("bias") or "LayerNorm.weight" in name else config["weight_decay"]
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                              for (lr, decay), params in groups.items()], eps=1e-8)


def train_weights(rows, mode, device):
    if mode == "none":
        return None
    counts = Counter(row[2] for row in rows)
    if set(counts) != set(range(len(LABELS))):
        raise ValueError("Balanced loss requires every class in train")
    return torch.tensor([len(rows) / (len(LABELS) * counts[i]) for i in range(len(LABELS))], device=device)


def evaluate(model, batches, device):
    model.eval()
    rows = []
    loss_sum = 0.
    with torch.no_grad():
        for ids, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            probabilities = logits.softmax(-1).cpu().tolist()
            for sid, truth, probs in zip(ids, labels.cpu().tolist(), probabilities, strict=True):
                rows.append({"sample_id": sid, "error_type": LABELS[truth],
                             "prediction": LABELS[max(range(len(LABELS)), key=lambda i: probs[i])],
                             **{f"p_{label}": probs[i] for i, label in enumerate(LABELS)}})
    result = scores([r["error_type"] for r in rows], [r["prediction"] for r in rows])
    result.update(loss=loss_sum / len(rows), support=len(rows),
                  prediction_counts={label: sum(r["prediction"] == label for r in rows) for label in LABELS})
    return result, rows


def write_predictions(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_history(directory, history):
    write_json(directory / "history.json", history)
    rows = []
    for entry in history:
        row = {"epoch": entry["epoch"], "elapsed_seconds": entry["elapsed_seconds"],
               "mean_batch_training_objective": entry["mean_batch_training_objective"]}
        for part in ("train", "validation"):
            for key in ("loss", "accuracy", "macro_f1"):
                row[f"{part}_{key}"] = entry[part][key]
            for label in LABELS:
                row[f"{part}_recall_{label}"] = entry[part]["per_label"][label]["recall"]
                row[f"{part}_predicted_{label}"] = entry[part]["prediction_counts"][label]
        rows.append(row)
    with (directory / "history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(directory, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    epochs = [row["epoch"] for row in history]
    for axis, key, title in zip(axes, ("loss", "accuracy", "macro_f1"),
                               ("Unweighted cross-entropy", "Accuracy", "Macro-F1"), strict=True):
        for part in ("train", "validation"):
            axis.plot(epochs, [row[part][key] for row in history], marker=".", label=part)
        axis.set(xlabel="Epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(figure)


def variants(config):
    for mode, encoder_lr, head_lr in itertools.product(
            config["readouts"], config["encoder_learning_rates"], config["head_learning_rates"]):
        yield {"readout": mode, "encoder_lr": encoder_lr, "head_lr": head_lr}


def run_name(variant, seed):
    return f"{variant['readout']}_enc{variant['encoder_lr']:g}_head{variant['head_lr']:g}_seed{seed}"


def save_best(directory, model, tokenizer, variant):
    directory.mkdir(exist_ok=True)
    model.base.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write_json(directory / "readout.json", {**variant, "labels": list(LABELS)})


def load_best(directory, device):
    info = read_json(directory / "readout.json")
    if info["labels"] != list(LABELS):
        raise ValueError("Checkpoint label order differs")
    tokenizer = BertTokenizerFast.from_pretrained(directory, local_files_only=True)
    base = BertForSequenceClassification.from_pretrained(directory, local_files_only=True)
    if base.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Checkpoint label mapping differs")
    start, end = tokenizer.convert_tokens_to_ids(["[SPAN_START]", "[SPAN_END]"])
    return BertReadout(base, info["readout"], start, end).to(device), tokenizer


def save_last(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_one(directory, variant, seed, config, source, tokenizer, data, device, env):
    directory.mkdir(exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(source, tokenizer, config, variant["readout"], device)
    optimizer = optimizer_for(model, config, variant)
    steps = math.ceil(len(data["train"]) / config["batch_size"]) * config["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * steps), steps)
    weights = train_weights(data["train"], config["class_weight"], device)
    train_eval = loader(data["train"], tokenizer, config["batch_size"])
    val_eval = loader(data["validation"], tokenizer, config["batch_size"])
    history, best, reference, bad_epochs, best_epoch = [], -1., -1., 0, 0
    elapsed = 0.
    last = directory / "last.pt"
    if last.exists():
        # This file is written by this program into the user's experiment directory.
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["environment"] != env:
            raise ValueError("Resume requires the same device and software environment")
        model.base.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, reference = state["history"], state["best"], state["reference"]
        bad_epochs, best_epoch, elapsed = state["bad_epochs"], state["best_epoch"], state["elapsed"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.perf_counter()
    stopped = bad_epochs >= config["patience"]
    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        if stopped:
            break
        model.train()
        objective_sum = 0.
        batches = loader(data["train"], tokenizer, config["batch_size"], seed + epoch)
        for _, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(**batch).logits, labels, weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach())
        train_metrics, _ = evaluate(model, train_eval, device)
        val_metrics, predictions = evaluate(model, val_eval, device)
        f1 = val_metrics["macro_f1"]
        if f1 > best:
            best, best_epoch = f1, epoch
            save_best(directory / "best", model, tokenizer, variant)
            write_predictions(directory / "validation_predictions.csv", predictions)
            write_json(directory / "best_metrics.json", {"epoch": epoch, "train": train_metrics,
                                                        "validation": val_metrics})
        if f1 > reference + config["min_delta"]:
            reference, bad_epochs = f1, 0
        else:
            bad_epochs += 1
        duration = elapsed + time.perf_counter() - started
        history.append({"epoch": epoch, "mean_batch_training_objective": objective_sum / len(batches),
                        "train": train_metrics, "validation": val_metrics, "elapsed_seconds": duration})
        save_last(last, {"model": model.base.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "history": history, "best": best,
                        "reference": reference, "bad_epochs": bad_epochs, "best_epoch": best_epoch,
                        "elapsed": duration, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                        "environment": env})
        write_history(directory, history)
        print(f"{directory.name} epoch={epoch} train_F1={train_metrics['macro_f1']:.4f} "
              f"val_F1={f1:.4f} best={best:.4f} patience={bad_epochs}/{config['patience']}", flush=True)
        stopped = bad_epochs >= config["patience"]
    write_history(directory, history)
    plot_history(directory, history)
    # Verify checkpoint serialization reproduces every saved validation prediction.
    saved_model, saved_tokenizer = load_best(directory / "best", device)
    replay_metrics, replay = evaluate(saved_model, loader(data["validation"], saved_tokenizer, config["batch_size"]), device)
    with (directory / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected = [(r["sample_id"], r["prediction"]) for r in csv.DictReader(stream)]
    if expected != [(r["sample_id"], r["prediction"]) for r in replay]:
        raise RuntimeError("Saved checkpoint does not reproduce validation predictions")
    summary = {"variant": variant, "seed": seed, "epochs_completed": len(history),
               "best_epoch": best_epoch, "stop_reason": "early_stopping" if stopped else "max_epochs",
               "best": read_json(directory / "best_metrics.json"), "test_scored": False,
               "checkpoint_hashes": tree_hashes(directory / "best"),
               "validation_prediction_sha256": digest(directory / "validation_predictions.csv"),
               "checkpoint_replay_macro_f1": replay_metrics["macro_f1"]}
    write_json(directory / "summary.json", summary)
    return summary


def check(config):
    device = runtime(config)
    source, tokenizer = source_and_tokenizer(config)
    data, stats = load_rows(config, tokenizer, ("train", "validation"))
    torch.manual_seed(config["seeds"][0])
    checks = {}
    for mode in config["readouts"]:
        model = make_model(source, tokenizer, config, mode, device).eval()
        _, batch = next(iter(loader(data["train"][:2], tokenizer, 2)))
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite preflight loss")
        target_mask(batch["input_ids"], batch["attention_mask"], model.start_id, model.end_id)
        checks[mode] = {"parameters": sum(p.numel() for p in model.parameters()), "finite_forward": True}
    print(json.dumps({"environment": environment(device), "data": stats, "checks": checks,
                      "planned_runs": len(list(variants(config))) * len(config["seeds"]),
                      "test_loaded": False, "training_started": False}, indent=2), flush=True)


def train(config, output, resume=False):
    device = runtime(config)
    source, tokenizer = source_and_tokenizer(config)
    env = environment(device)
    manifest = {"config": config, "source": source, "inputs": input_identity(config, source),
                "code": code_hashes(), "environment": env, "test_scored": False}
    if output.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume with identical config or a new output directory")
        if read_json(output / "manifest.json") != manifest:
            raise ValueError("Cannot resume: config, inputs, code or environment changed")
        if (output / "selection.json").exists():
            raise ValueError("Experiment already locked; use a new output for new training")
    else:
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest)
        write_json(output / "config.json", config)
    data, stats = load_rows(config, tokenizer, ("train", "validation"))
    write_json(output / "data_audit.json", stats)
    for variant in variants(config):
        for seed in config["seeds"]:
            directory = output / run_name(variant, seed)
            if (directory / "summary.json").exists():
                validate_run(directory, variant, seed)
                print(f"Completed, skipping {directory.name}", flush=True)
                continue
            train_one(directory, variant, seed, config, source, tokenizer, data, device, env)
    print("Training complete. Inspect histories, then run select to lock the validation choice.", flush=True)


def validate_run(directory, variant, seed):
    summary = read_json(directory / "summary.json")
    if summary["variant"] != variant or summary["seed"] != seed:
        raise ValueError("Run configuration mismatch")
    if tree_hashes(directory / "best") != summary["checkpoint_hashes"]:
        raise ValueError(f"Checkpoint changed: {directory}")
    if digest(directory / "validation_predictions.csv") != summary["validation_prediction_sha256"]:
        raise ValueError(f"Validation predictions changed: {directory}")
    return summary


def mean_std(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def select(output):
    if (output / "selection.json").exists():
        raise FileExistsError("Selection already locked")
    manifest = read_json(output / "manifest.json")
    config = manifest["config"]
    if code_hashes() != manifest["code"]:
        raise ValueError("Code changed since training; restore the experiment version before selection")
    ranking = []
    for variant in variants(config):
        runs = [validate_run(output / run_name(variant, seed), variant, seed) for seed in config["seeds"]]
        values = [run["best"]["validation"]["macro_f1"] for run in runs]
        ranking.append({"variant": variant, "validation_macro_f1": mean_std(values),
                        "runs": [run_name(variant, seed) for seed in config["seeds"]]})
    # Stable order resolves ties; seed order is declared before training.
    chosen = max(ranking, key=lambda row: row["validation_macro_f1"]["mean"])
    lock = {"rule": "highest mean best-checkpoint validation macro-F1 across all declared seeds; grid order breaks ties",
            "ranking": ranking, "selected": chosen, "primary_seed": config["seeds"][0],
            "primary_run": chosen["runs"][0], "test_scored": False,
            "manifest_sha256": digest(output / "manifest.json"), "code": manifest["code"],
            "checkpoints": {name: tree_hashes(output / name / "best") for name in chosen["runs"]}}
    write_json(output / "selection.json", lock)
    (output / "selection.sha256").write_text(digest(output / "selection.json") + "\n", encoding="utf-8")
    print(json.dumps(chosen, indent=2), flush=True)
    print("Selection locked. Test all declared seeds; primary seed remains the first configured seed.", flush=True)


def test(selection_path, output):
    root = selection_path.parent
    expected = selection_path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if digest(selection_path) != expected:
        raise ValueError("Selection lock changed")
    lock = read_json(selection_path)
    manifest_path = root / "manifest.json"
    if digest(manifest_path) != lock["manifest_sha256"] or code_hashes() != lock["code"]:
        raise ValueError("Manifest/code changed since training; restore the experiment version")
    manifest = read_json(manifest_path)
    config = manifest["config"]
    for key in ("samples", "split", "model_source"):
        if digest(config[key]) != manifest["inputs"][key]:
            raise ValueError(f"Input changed: {key}")
    for name, hashes in lock["checkpoints"].items():
        if tree_hashes(root / name / "best") != hashes:
            raise ValueError(f"Locked checkpoint changed: {name}")
    # Selection/checkpoint verification happens before any test tokenization or scoring.
    if output.exists():
        raise FileExistsError("Test output exists; keep the recorded result")
    device = runtime(config)
    tokenizer = BertTokenizerFast.from_pretrained(root / lock["primary_run"] / "best", local_files_only=True)
    data, audit = load_rows(config, tokenizer, ("test",))
    output.mkdir(parents=True)
    write_json(output / "test_started.json", {"selection_sha256": expected, "audit": audit,
                                             "environment": environment(device)})
    results = []
    for name in lock["selected"]["runs"]:
        model, run_tokenizer = load_best(root / name / "best", device)
        if run_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Selected seed tokenizers differ")
        metrics, predictions = evaluate(model, loader(data["test"], tokenizer, config["batch_size"]), device)
        write_predictions(output / f"{name}_test_predictions.csv", predictions)
        write_json(output / f"{name}_test_metrics.json", metrics)
        results.append({"run": name, "seed": read_json(root / name / "summary.json")["seed"], "metrics": metrics})
        print(f"{name} test macro-F1={metrics['macro_f1']:.4f}", flush=True)
    summary = {"selection_sha256": expected, "primary_run": lock["primary_run"], "runs": results,
               "test_macro_f1": mean_std([r["metrics"]["macro_f1"] for r in results]),
               "test_accuracy": mean_std([r["metrics"]["accuracy"] for r in results]),
               "scope": "Same locked split; no train+validation refit; seed std is not a group-bootstrap CI",
               "test_scored": True}
    write_json(output / "summary.json", summary)
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "seed", "label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for result in results:
            for label, metrics in result["metrics"]["per_label"].items():
                writer.writerow({"run": result["run"], "seed": result["seed"], "label": label, **metrics})
    print(json.dumps({"test_macro_f1": summary["test_macro_f1"], "test_accuracy": summary["test_accuracy"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("check", help="Validate train/validation inputs and run two-example forwards; no training")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    training = sub.add_parser("train", help="Train grid, evaluate train/validation, save best and resumable checkpoints")
    training.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    selection = sub.add_parser("select", help="Lock configuration using all declared seeds on validation only")
    selection.add_argument("--experiment", type=Path, required=True)
    testing = sub.add_parser("test", help="Score locked checkpoints on test; never trains")
    testing.add_argument("--selection", type=Path, required=True)
    testing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(load_config(args.config))
    elif args.command == "train":
        train(load_config(args.config), resolve(args.output), args.resume)
    elif args.command == "select":
        select(resolve(args.experiment))
    else:
        test(resolve(args.selection), resolve(args.output))


if __name__ == "__main__":
    main()
