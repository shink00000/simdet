from itertools import product

import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import box_convert

from .backbones import BACKBONES
from .necks import FPN
from .losses import FocalLoss
from .postprocesses import MultiLabelNMS


class RetinaHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_stacks: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.reg_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(n_stacks)
        ])
        self.cls_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(n_stacks)
        ])
        self.reg_top = nn.Conv2d(in_channels, 9*4, kernel_size=3, padding=1)
        self.cls_top = nn.Conv2d(in_channels, 9*n_classes, kernel_size=3, padding=1)

        self._init_weights()

    def forward(self, xs: list):
        reg_outs, cls_outs = [], []
        for x in xs:
            reg_out = self.reg_top(self.reg_convs(x))
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(reg_out.size(0), -1, 4)
            reg_outs.append(reg_out)
            cls_out = self.cls_top(self.cls_convs(x))
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(cls_out.size(0), -1, self.n_classes)
            cls_outs.append(cls_out)
        reg_outs = torch.cat(reg_outs, dim=1)
        cls_outs = torch.cat(cls_outs, dim=1)

        return reg_outs, cls_outs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    if 'cls_top' in name:
                        nn.init.constant_(m.bias, np.log((1 - 0.01) / 0.01))
                    else:
                        nn.init.constant_(m.bias, 0.0)


class RetinaNet(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list):
        super().__init__()

        # layers
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.neck = FPN([self.backbone.C3, self.backbone.C4, self.backbone.C5], 256)
        self.head = RetinaHead(256, n_classes)

        # property
        H, W = input_size
        strides = [2**i for i in range(3, 8)]
        prior_boxes = []
        for stride in strides:
            for cy, cx in product(range(stride//2, H, stride), range(stride//2, W, stride)):
                for aspect in [0.5, 1.0, 2.0]:
                    for scale in [0, 1/3, 2/3]:
                        h = 4 * stride * pow(2, scale) * pow(aspect, 1/2)
                        w = 4 * stride * pow(2, scale) * pow(1/aspect, 1/2)
                        prior_boxes.append([cx, cy, w, h])
        self.register_buffer('prior_boxes', torch.Tensor(prior_boxes))

        # loss
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        self.cls_loss = FocalLoss(reduction='sum')

        # postprocess
        self.postprocess = MultiLabelNMS()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.head(x)
        return outs

    def parameters(self, cfg):
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
        reg_outs, cls_outs = outputs
        reg_targets, cls_targets = targets

        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        N = pos_mask.sum()

        if N > 0:
            reg_loss = self.reg_loss(reg_outs[pos_mask], reg_targets[pos_mask]) / N
            cls_loss = self.cls_loss(cls_outs[pos_mask + neg_mask], cls_targets[pos_mask + neg_mask]) / N
            loss = reg_loss + cls_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outputs: tuple) -> tuple:
        reg_outs, cls_outs = outputs
        bboxes = box_convert(self._delta2bbox(reg_outs), 'cxcywh', 'xyxy')
        scores = cls_outs.sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    def _delta2bbox(self, reg_outs) -> torch.Tensor:
        cx = self.prior_boxes[..., 0] + 0.1 * reg_outs[..., 0] * self.prior_boxes[..., 2]
        cy = self.prior_boxes[..., 1] + 0.1 * reg_outs[..., 1] * self.prior_boxes[..., 3]
        w = self.prior_boxes[..., 2] * (0.2 * reg_outs[..., 2]).exp()
        h = self.prior_boxes[..., 3] * (0.2 * reg_outs[..., 3]).exp()

        return torch.stack([cx, cy, w, h], dim=-1)
