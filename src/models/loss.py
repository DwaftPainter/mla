"""Cross entropy accepts raw logits and integer class indices."""
from torch import nn


def create_loss():
    return nn.CrossEntropyLoss()
