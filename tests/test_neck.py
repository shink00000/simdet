import torch

from simdet.models.necks.fpn import FPN


def test_fpn():
    model = FPN([8, 16], 64)
    xs = [torch.rand(2, 8, 128, 128), torch.rand(2, 16, 64, 64)]
    ys = model(xs)
    for i in range(1, 6):
        assert ys[i-1].size(-1) == 128 / 2 ** (i-1)
        assert ys[i-1].size(1) == 64
