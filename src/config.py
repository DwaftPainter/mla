"""Read YAML; resolve relative paths against the project, not the shell cwd."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    for name, value in config["paths"].items():
        config["paths"][name] = str((PROJECT_ROOT / value).resolve())
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config["model"]["input_dim"] != 2 or config["model"]["num_classes"] != 2:
        raise ValueError("The ring dataset requires input_dim=2 and num_classes=2")
    return config
