import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    resnext50_32x4d
)


class ResNet(nn.Module):
    def __init__(self, depth: int, frozen_stages: int = -1, **kwargs):
        super().__init__()
        base = self.map_to_model(depth)(pretrained=True, **kwargs)
        for name, m in base.named_children():
            if 'avgpool' in name:
                break
            elif not (name.startswith('layer') and int(name[-1]) > frozen_stages):
                m.requires_grad_(False)
            else:
                for m_ in m.modules():
                    if isinstance(m_, nn.BatchNorm2d):
                        m_.requires_grad_(False)
            setattr(self, name, m)

        with torch.no_grad():
            x = torch.rand(2, 3, 32, 32)
            outs = self(x)
            for i, out in enumerate(outs, start=1):
                setattr(self, f'C{i}', out.size(1))

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5

    def map_to_model(self, depth):
        return {
            18: resnet18,
            50: resnet50,
        }[depth]

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNeXt(ResNet):
    def map_to_model(self, depth):
        return {
            50: resnext50_32x4d,
        }[depth]
