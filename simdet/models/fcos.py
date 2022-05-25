from itertools import product

import torch
import torch.nn as nn
import numpy as np

from .backbones import BACKBONES
from .necks import FPN
from .losses import IoULossWithDistance, FocalLoss
from .postprocesses import MultiLabelNMS


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

    def extra_repr(self) -> str:
        return f'scale={self.scale}'


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_stacks: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.reg_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(n_stacks)
        ])
        self.cls_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(n_stacks)
        ])
        self.reg_top = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.cls_top = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1)
        self.cnt_top = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

        self._init_weights()

    def forward(self, xs: list):
        reg_outs, cls_outs, cnt_outs = [], [], []
        for i, x in enumerate(xs):
            reg_out = self.scales[i](self.reg_top(self.reg_convs(x))).exp()
            reg_outs.append(reg_out.permute(0, 2, 3, 1).flatten(1, 2))
            cls_feat = self.cls_convs(x)
            cls_out = self.cls_top(cls_feat)
            cls_outs.append(cls_out.permute(0, 2, 3, 1).flatten(1, 2))
            cnt_out = self.cnt_top(cls_feat)
            cnt_outs.append(cnt_out.permute(0, 2, 3, 1).flatten(1, 2))
        reg_outs = torch.cat(reg_outs, dim=1)
        cls_outs = torch.cat(cls_outs, dim=1)
        cnt_outs = torch.cat(cnt_outs, dim=1)

        return reg_outs, cls_outs, cnt_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)


class FCOS(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list):
        super().__init__()

        # layers
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.neck = FPN([self.backbone.C3, self.backbone.C4, self.backbone.C5], 256)
        self.head = FCOSHead(256, n_classes)

        # property
        H, W = input_size
        strides = [2**i for i in range(3, 8)]
        all_points = []
        for stride in strides:
            points = [[x, y] for y, x in product(
                range(stride//2, H, stride), range(stride//2, W, stride)
            )]
            all_points.extend(points)
        self.register_buffer('all_points', torch.Tensor(all_points))

        # loss
        self.reg_loss = IoULossWithDistance(reduction='sum')
        self.cls_loss = FocalLoss(reduction='sum')
        self.cnt_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # postprocess
        self.postprocess = MultiLabelNMS()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.head(x)
        return outs

    def get_param_groups(self, cfg):
        base_lr = cfg['lr']
        param_groups = [
            {'params': [], 'lr': base_lr * 0.1 * 2, 'weight_decay': 0.0},
            {'params': [], 'lr': base_lr * 0.1},
            {'params': [], 'lr': base_lr * 2, 'weight_decay': 0.0},
            {'params': []},
        ]
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'backbone' in name:
                    no = 0 if p.ndim == 1 else 1
                else:
                    no = 2 if p.ndim == 1 else 3
                param_groups[no]['params'].append(p)

        return param_groups

    def loss(self, outputs: tuple, targets: tuple) -> torch.Tensor:
        reg_outs, cls_outs, cnt_outs = outputs
        reg_targets, cls_targets, cnt_targets = targets

        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        N = pos_mask.sum()

        if N > 0:
            reg_loss = self.reg_loss(reg_outs[pos_mask], reg_targets[pos_mask]) / N
            cnt_loss = self.cnt_loss(cnt_outs[pos_mask], cnt_targets[pos_mask]) / N
            cls_loss = self.cls_loss(cls_outs[pos_mask + neg_mask], cls_targets[pos_mask + neg_mask]) / N
            loss = reg_loss + cnt_loss + cls_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outputs: tuple) -> tuple:
        reg_outs, cls_outs, cnt_outs = outputs
        bboxes = self._distance2bbox(reg_outs)
        scores = cls_outs.sigmoid() * cnt_outs.sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    def _distance2bbox(self, reg_outs) -> torch.Tensor:
        xmin = self.all_points[..., 0] - reg_outs[..., 0]
        ymin = self.all_points[..., 1] - reg_outs[..., 1]
        xmax = self.all_points[..., 0] + reg_outs[..., 2]
        ymax = self.all_points[..., 1] + reg_outs[..., 3]

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
