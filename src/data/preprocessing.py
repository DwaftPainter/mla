"""Generate/load CSV, stratify before fitting normalization, and persist tensors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def generate_demo_data(path, settings, seed):
    """Create a deterministic demo CSV only when it does not already exist."""
    path = Path(path)
    if path.exists():
        return
    count = settings["samples"]
    if count < 20 or count % 2:
        raise ValueError("samples must be even and at least 20")
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], count // 2)
    angles = rng.uniform(0, 2 * np.pi, count)
    radii = np.where(labels == 0, settings["inner_radius"], settings["outer_radius"])
    features = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    features += rng.normal(0, settings["noise"], features.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((features, labels)), delimiter=",",
               header="x1,x2,label", comments="", fmt="%.10f")


def load_raw(path):
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if rows.shape[1] != 3 or not np.isfinite(rows).all():
        raise ValueError("CSV must contain finite x1,x2,label columns")
    if set(rows[:, 2]) != {0.0, 1.0}:
        raise ValueError("CSV labels must contain exactly classes 0 and 1")
    # Reject duplicate inputs rather than letting the same example cross splits.
    if len(np.unique(rows[:, :2], axis=0)) != len(rows):
        raise ValueError("CSV contains duplicate feature rows; clean them before training")
    return rows[:, :2], rows[:, 2].astype(np.int64)


def transform(values, statistics):
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features[None, :]
    mean = np.asarray(statistics["mean"])
    std = np.asarray(statistics["std"])
    if features.ndim != 2 or features.shape[1] != len(mean):
        raise ValueError(f"Expected inputs with {len(mean)} features")
    if not np.isfinite(features).all():
        raise ValueError("Input features must be finite")
    normalized = ((features - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized features exceed float32 range")
    return torch.from_numpy(normalized)


def prepare_data(config):
    settings, paths = config["dataset"], config["paths"]
    generate_demo_data(paths["raw"], settings, config["seed"])
    features, targets = load_raw(paths["raw"])
    train_fraction = settings["train_fraction"]
    validation_fraction = settings["validation_fraction"]
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and leave a nonempty test fraction")
    rng = np.random.default_rng(config["seed"])
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(targets == label))
        n_train = int(len(indices) * train_fraction)
        n_validation = int(len(indices) * validation_fraction)
        parts = np.split(indices, [n_train, n_train + n_validation])
        if any(len(part) == 0 for part in parts):
            raise ValueError("Each split must have at least one example of each class")
        for name, part in zip(splits, parts):
            splits[name].extend(part.tolist())
    splits = {name: rng.permutation(ids) for name, ids in splits.items()}
    training_features = features[splits["train"]]
    std = training_features.std(axis=0)
    metadata = {
        "mean": training_features.mean(axis=0).tolist(),
        "std": np.where(std > 0, std, 1.0).tolist(),
        "class_names": settings["class_names"],
        "raw_sha256": hashlib.sha256(Path(paths["raw"]).read_bytes()).hexdigest(),
        "seed": config["seed"],
    }
    processed = Path(paths["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "normalization.json").write_text(json.dumps(metadata, indent=2))
    for name, ids in splits.items():
        directory = Path(paths["split"]) / name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"features": transform(features[ids], metadata),
                    "targets": torch.tensor(targets[ids], dtype=torch.long),
                    "ids": torch.tensor(ids, dtype=torch.long),
                    "preprocessing": metadata}, directory / "data.pt")
    return metadata
