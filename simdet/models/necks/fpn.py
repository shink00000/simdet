from collections import deque
import torch.nn as nn


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
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(self.n_feats)])

        self._init_weights()

    def forward(self, xs: list) -> list:
        # resample
        xs = deque(xs, maxlen=self.n_feats)
        for resample in self.resamples:
            xs.append(resample(xs[-1]))

        # fpn
        outs = [self.lateral_convs[i](xs[i]) for i in range(self.n_feats)]
        for i in range(self.n_feats-2, -1, -1):
            outs[i] = outs[i] + self.up(outs[i+1])
        outs = [self.convs[i](outs[i]) for i in range(self.n_feats)]

        return outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
