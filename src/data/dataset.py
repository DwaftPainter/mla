"""The only interface training needs to access prepared examples."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class PointDataset(Dataset):
    def __init__(self, config, split):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        path = Path(config["paths"]["split"]) / split / "data.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run training to prepare data first.")
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.features = data["features"]
        self.targets = data["targets"]
        self.ids = data["ids"]
        self.preprocessing = data["preprocessing"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def make_loader(config, split):
    return DataLoader(
        PointDataset(config, split), batch_size=config["training"]["batch_size"],
        shuffle=split == "train", num_workers=config["training"]["num_workers"],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
