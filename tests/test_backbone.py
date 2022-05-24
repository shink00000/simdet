import torch

from simdet.models.backbones.resnet import ResNet


def test_resnet():
    model = ResNet(depth=18)
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    for i in range(1, 6):
        assert y[i-1].size(-1) == 32 / 2 ** i
        assert hasattr(model, f'C{i}')
