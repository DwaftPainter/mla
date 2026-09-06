"""Shared model restoration, independent of training and raw data."""
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch

from src.models.model import Model


def load_checkpoint(path, device="cpu"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run training first.")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Model(**checkpoint["configuration"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
