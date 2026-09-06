"""Evaluate the selected checkpoint ONLY on the held-out test partition."""
import argparse
import json
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.config import DEFAULT_CONFIG, load_config
from src.data.dataset import make_loader
from src.evaluation.metrics import score
from src.models.loss import create_loss


def evaluate(checkpoint_path, split_dir=None):
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["configuration"]
    if split_dir is not None:
        config["paths"]["split"] = str(Path(split_dir).resolve())
    loader = make_loader(config, "test")
    if (loader.dataset.preprocessing != checkpoint["preprocessing"]
            or loader.dataset.ids.tolist() != checkpoint["test_ids"]):
        raise ValueError("Prepared test split does not match this checkpoint; restore its original split")
    return score(model, loader, create_loss(), "cpu", config["model"]["num_classes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-dir", type=Path, help="Override location after moving the project")
    args = parser.parse_args()
    checkpoint = args.checkpoint or load_config(args.config)["paths"]["checkpoint"]
    print(json.dumps(evaluate(checkpoint, args.split_dir), indent=2))


if __name__ == "__main__":
    main()
