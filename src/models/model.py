"""A small multilayer perceptron: [batch, 2] -> [batch, 2] logits."""
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)
