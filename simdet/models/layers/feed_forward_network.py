import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))

        return out
