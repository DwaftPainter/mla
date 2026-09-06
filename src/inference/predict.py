"""Load learned weights and predict without training or reading the dataset."""
import argparse
import json
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.config import DEFAULT_CONFIG, load_config
from src.data.preprocessing import transform


def predict(checkpoint_path, features):
    model, checkpoint = load_checkpoint(checkpoint_path)
    inputs = transform(features, checkpoint["preprocessing"])
    if inputs.shape[0] != 1:
        raise ValueError("predict expects one example with two features")
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    class_id = probabilities.argmax().item()
    return {"class_id": class_id,
            "class_name": checkpoint["preprocessing"]["class_names"][class_id],
            "confidence": probabilities[class_id].item(),
            "probabilities": probabilities.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--features", nargs=2, type=float, default=[0.7, 0.0], metavar=("X1", "X2"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or load_config(args.config)["paths"]["checkpoint"]
    print(json.dumps(predict(checkpoint, args.features), indent=2))


if __name__ == "__main__":
    main()
