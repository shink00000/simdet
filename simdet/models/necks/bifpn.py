from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.norm(x)
        out = self.act(x)

        return out


class BiFPNBlock(nn.Module):
    eps = 1e-4

    def __init__(self, in_channels: list, out_channels: list, n_feats: int):
        assert len(in_channels) == len(out_channels) == n_feats

        super().__init__()
        self.n_feats = n_feats
        id = nn.Identity()
        self.x2h_convs = nn.ModuleList([*[nn.Conv2d(in_channels[i], out_channels[i], 1) for i in range(n_feats-1)], id])
        self.h2y_convs = nn.ModuleList([id, *[nn.Conv2d(out_channels[i], out_channels[i], 1) for i in range(1, n_feats)]])
        self.x2y_convs = nn.ModuleList([None, *[nn.Conv2d(in_channels[i], out_channels[i], 1) for i in range(1, n_feats-1)], None])
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dw = nn.Upsample(scale_factor=1/2, mode='nearest')
        self.convs = nn.ModuleList([SeparableConv2d(out_channels[i], out_channels[i], 3, padding=1) for i in range(n_feats)])

        self.w2 = nn.Parameter(torch.ones(n_feats, 2))
        self.w3 = nn.Parameter(torch.ones(n_feats-2, 3))

    def forward(self, xs: list) -> list:
        # top down pathway
        hs = [self.x2h_convs[i](xs[i]) for i in range(self.n_feats)]
        w2 = F.relu(self.w2, inplace=False)
        for i in range(self.n_feats-2, -1, -1):
            w = w2[i]
            hs[i] = w[0] * hs[i] + w[1] * self.up(hs[i+1])
            hs[i] /= (w.sum() + self.eps)

        # bottom up pathway
        ys = [self.h2y_convs[i](hs[i]) for i in range(self.n_feats)]
        w3 = F.relu(self.w3, inplace=False)
        for i in range(1, self.n_feats):
            if i < self.n_feats-1:
                w = w3[i-1]
                ys[i] = w[0] * ys[i] + w[1] * self.dw(ys[i-1]) + w[2] * self.x2y_convs[i](xs[i])
            else:
                w = w2[i]
                ys[i] = w[0] * ys[i] + w[1] * self.dw(ys[i-1])
            ys[i] /= (w.sum() + self.eps)

        outs = [self.convs[i](ys[i]) for i in range(self.n_feats)]
        return outs


class BiFPN(nn.Module):
    def __init__(self, in_channels: list, out_channels: int, n_blocks: int):
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
        out_channels = [out_channels] * self.n_feats

        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.blocks.append(BiFPNBlock(in_channels, out_channels, self.n_feats))
            in_channels = out_channels

        self._init_weights()

    def forward(self, xs: list) -> list:
        # resample
        xs = deque(xs, maxlen=self.n_feats)
        for resample in self.resamples:
            xs.append(resample(xs[-1]))

        # bifpn
        for block in self.blocks:
            xs = block(xs)

        return xs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
