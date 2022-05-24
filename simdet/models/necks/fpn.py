from collections import deque
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels: list, out_channels: int):
        super().__init__()
        self.n_feats = 5
        self.resamples = nn.ModuleList([])
        for i in range(self.n_feats-len(in_channels)):
            if i == 0:
                m = nn.Conv2d(in_channels[-1], out_channels, 3, 2, 1)
            else:
                m = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 2, 1)
                )
            self.resamples.append(m)
            in_channels.append(out_channels)
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels, 1) for i in range(self.n_feats)])
        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(self.n_feats)])

    def forward(self, xs: list):
        xs = deque(xs, maxlen=self.n_feats)
        for resample in self.resamples:
            xs.append(resample(xs[-1]))

        outs = [self.lateral_convs[i](xs[i]) for i in range(self.n_feats)]
        for i in range(self.n_feats-2, -1, -1):
            outs[i] = outs[i] + F.interpolate(outs[i+1], scale_factor=2, mode='nearest')
        outs = [self.convs[i](outs[i]) for i in range(self.n_feats)]

        return outs


if __name__ == '__main__':
    import torch
    m = FPN([8, 16, 32], 64)
    xs = [torch.rand(2, 8, 128, 128), torch.rand(2, 16, 64, 64), torch.rand(2, 32, 32, 32)]
    xs = m(xs)
    for x in xs:
        print(x.shape)
    print(m)
