"""Run with: python -m src.training.train"""
import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import DEFAULT_CONFIG, load_config
from src.data.dataset import make_loader
from src.data.preprocessing import prepare_data
from src.evaluation.metrics import score
from src.models.loss import create_loss
from src.models.model import Model

LOGGER = logging.getLogger(__name__)


def train(config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training"]["num_threads"])
    preprocessing = prepare_data(config)
    train_loader = make_loader(config, "train")
    validation_loader = make_loader(config, "validation")
    # Record the held-out partition identity without scoring it during training.
    test_ids = make_loader(config, "test").dataset.ids.tolist()
    device = torch.device(config["training"]["device"])
    model = Model(**config["model"]).to(device)
    criterion = create_loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    checkpoint_path = Path(config["paths"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, history = float("inf"), []
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear gradients left by the previous batch.
            logits = model(inputs)  # Forward pass through the current weights.
            loss = criterion(logits, targets)
            loss.backward()  # Backpropagation computes gradients; weights stay unchanged.
            optimizer.step()  # Adam changes the trainable weights and biases HERE.
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            count += targets.size(0)
        validation = score(model, validation_loader, criterion, device, config["model"]["num_classes"])
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "train_accuracy": correct / count,
               "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"]}
        history.append(row)
        LOGGER.info("epoch %02d train loss=%.4f acc=%.3f | validation loss=%.4f acc=%.3f",
                    epoch, row["train_loss"], row["train_accuracy"],
                    row["validation_loss"], row["validation_accuracy"])
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "validation_metric": best_loss,
                        "validation_metric_name": "loss", "validation_metrics": validation,
                        "configuration": config, "preprocessing": preprocessing,
                        "test_ids": test_ids}, checkpoint_path)
    if not checkpoint_path.exists() or best_loss == float("inf"):
        raise RuntimeError("No finite validation loss; no checkpoint saved for this run")
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path, _ = train(load_config(args.config))
    LOGGER.info("Best checkpoint saved to %s", path)
    LOGGER.info("Run python -m src.evaluation.evaluate for final test metrics.")


if __name__ == "__main__":
    main()
