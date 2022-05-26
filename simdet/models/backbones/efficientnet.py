import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7
)


class EfficientNet(nn.Module):
    def __init__(self, size: str, frozen_stages: int = -1, **kwargs):
        super().__init__()
        base = self.map_to_model(size)(pretrained=True, **kwargs)
        self.features = base.features[:8]
        for i, m in enumerate(self.features):
            if i <= frozen_stages:
                m.requires_grad_(False)
            else:
                break

        with torch.no_grad():
            x = torch.rand(2, 3, 32, 32)
            outs = self(x)
            for i, out in enumerate(outs, start=1):
                setattr(self, f'C{i}', out.size(1))

    def forward(self, x):
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in (1, 2, 3, 5, 7):
                outs.append(x)

        return outs

    def map_to_model(self, size):
        return {
            'b0': efficientnet_b0,
            'b1': efficientnet_b1,
            'b2': efficientnet_b2,
            'b3': efficientnet_b3,
            'b4': efficientnet_b4,
            'b5': efficientnet_b5,
            'b6': efficientnet_b6,
            'b7': efficientnet_b7,
        }[size]
