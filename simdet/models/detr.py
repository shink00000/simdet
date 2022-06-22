import torch
import torch.nn as nn
import numpy as np
from torchvision.ops.boxes import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment

from .layers import nchw_to_nlc, MultiheadAttention, FeedForwardNetwork, DropPath, SineEncoding
from .backbones import BACKBONES
from .losses import GIoULoss, FocalLoss
from .postprocesses import MultiLabelBasicProcess


class DETREncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, n_heads)
        self.drop1 = DropPath(drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim)
        self.drop2 = DropPath(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        shortcut = x
        q = k = x + x_pe
        v = x
        x = self.attn(q, k, v)
        x = self.norm1(x + self.drop1(shortcut))

        shortcut = x
        x = self.ffn(x)
        out = self.norm2(x + self.drop2(shortcut))

        return out


class DETREncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.layers = nn.ModuleList([
            DETREncoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, x_pe)

        return x


class DETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, n_heads)
        self.drop1 = DropPath(drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadAttention(embed_dim, n_heads)
        self.drop2 = DropPath(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim)
        self.drop3 = DropPath(drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, con_query: torch.Tensor, x: torch.Tensor, x_pe: torch.Tensor, object_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            con_query (torch.Tensor): (N, n_objs, C)
            x (torch.Tensor): (N, L, C)
            x_pe (torch.Tensor): (1, L, C)
            object_query (torch.Tensor): (N, n_objs, C)

        Returns:
            torch.Tensor: (N, n_objs, C)
        """
        # self attention
        shortcut = con_query
        q = k = con_query + object_query
        v = con_query
        con_query = self.self_attn(q, k, v)
        con_query = self.norm1(con_query + self.drop1(shortcut))

        # cross attention
        shortcut = con_query
        q = con_query + object_query
        k = x + x_pe
        v = x
        con_query = self.cross_attn(q, k, v)
        con_query = self.norm2(con_query + self.drop2(shortcut))

        # ffn
        shortcut = con_query
        con_query = self.ffn(con_query)
        out = self.norm3(con_query + self.drop3(shortcut))

        return out


class DETRDecoder(nn.Module):
    def __init__(self, n_objs, n_layers, embed_dim, n_heads, drop_rates, n_classes):
        super().__init__()
        self.object_query = nn.Parameter(torch.zeros(n_objs, embed_dim))
        self.layers = nn.ModuleList([
            DETRDecoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.reg_top = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4)
        )
        self.cls_top = nn.Linear(embed_dim, n_classes)

        nn.init.normal_(self.object_query)

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        outs = []
        object_query = self.object_query.unsqueeze(0).repeat(x.size(0), 1, 1)
        con_query = torch.zeros_like(object_query)
        for layer in self.layers:
            con_query = layer(con_query, x, x_pe, object_query)
            outs.append(self.norm(con_query))
            object_query = object_query.detach()
        outs = torch.stack(outs)
        reg_outs = self.reg_top(outs).sigmoid()
        cls_outs = self.cls_top(outs)

        return reg_outs, cls_outs


class DETRHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_objs: int = 100, n_encoders: int = 6, n_decoders: int = 6):
        super().__init__()
        embed_dim = 256
        n_heads = 8
        dprs = torch.linspace(0, 0.1, n_encoders + n_decoders).tolist()

        self.projection = nn.Conv2d(in_channels, embed_dim, 1)
        self.pos_encoding = SineEncoding(embed_dim, 2)
        self.x_pe = None
        self.encoder = DETREncoder(n_encoders, embed_dim, n_heads, dprs[:n_encoders])
        self.decoder = DETRDecoder(n_objs, n_decoders, embed_dim, n_heads, dprs[n_encoders:], n_classes)

        self._init_weights()

    def forward(self, xs: list):
        x = self.projection(xs[-1])
        n, _, h, w = x.shape
        x = nchw_to_nlc(x)
        if self.x_pe is None:
            pos_x = torch.arange(w, device=x.device).view(1, -1).repeat(h, 1).flatten() / w
            pos_y = torch.arange(h, device=x.device).view(-1, 1).repeat(1, w).flatten() / h
            pos = torch.stack([pos_x, pos_y], dim=-1).unsqueeze(0)
            self.x_pe = self.pos_encoding(pos)
        x_pe = self.x_pe.repeat(n, 1, 1)
        x = self.encoder(x, x_pe)
        reg_outs, cls_outs = self.decoder(x, x_pe)

        return reg_outs, cls_outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(self.decoder.cls_top.bias, -np.log((1 - 0.01) / 0.01))


class DETR(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, n_objs: int = 100,
                 lmd_l1: int = 5, lmd_iou: int = 2):
        super().__init__()

        # layers
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.head = DETRHead(self.backbone.C5, n_classes, n_objs)

        # property
        self.H, self.W = input_size
        self.lmd_l1 = lmd_l1
        self.lmd_iou = lmd_iou

        # loss
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        self.iou_loss = GIoULoss(reduction='sum')
        self.cls_loss = FocalLoss(reduction='sum')

        # postprocess
        self.postprocess = MultiLabelBasicProcess()

    def forward(self, x):
        x = self.backbone(x)
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
        loss = 0
        for i in range(outputs[0].size(0)):
            reg_outs, cls_outs = outputs
            outputs_ = (reg_outs[i], cls_outs[i])
            loss += self._loss_single(outputs_, targets)

        return loss

    def _loss_single(self, outputs: tuple, targets: tuple) -> torch.Tensor:
        reg_outs, cls_outs = outputs
        reg_targets, cls_targets, pos_targets = targets

        reg_outs_, reg_targets_ = [], []
        cls_outs_, cls_targets_ = [], []
        for i in range(reg_outs.size(0)):
            reg_out = reg_outs[i].detach()
            cls_out = cls_outs[i].detach()
            reg_target = reg_targets[i][:pos_targets[i]]
            cls_target = cls_targets[i][:pos_targets[i]]
            cost = -cls_out[:, cls_target-1]
            cost += self.lmd_l1 * torch.cdist(reg_out, reg_target, p=1)
            cost += self.lmd_iou * (1 - generalized_box_iou(self._to_xyxy(reg_out), self._to_xyxy(reg_target)))

            row_ind, col_ind = linear_sum_assignment(cost.cpu())

            reg_outs_.append(reg_outs[i][row_ind])
            reg_targets_.append(reg_target[col_ind])
            cls_outs_.append(cls_outs[i])
            new_cls_target = torch.zeros(cls_out.size(0), dtype=torch.long, device=cls_target.device)
            new_cls_target[row_ind] = cls_target[col_ind]
            cls_targets_.append(new_cls_target)

        reg_outs, reg_targets = torch.cat(reg_outs_), torch.cat(reg_targets_)
        cls_outs, cls_targets = torch.cat(cls_outs_), torch.cat(cls_targets_)

        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0
        N = pos_mask.sum()

        if N > 0:
            cls_loss = self.cls_loss(cls_outs, cls_targets) / N
            reg_loss = self.reg_loss(reg_outs, reg_targets) / N
            iou_loss = self.iou_loss(self._to_xyxy(reg_outs), self._to_xyxy(reg_targets)) / N
            loss = cls_loss + self.lmd_l1 * reg_loss + self.lmd_iou * iou_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outputs: tuple) -> tuple:
        reg_outs, cls_outs = [output[-1] for output in outputs]
        bboxes = self._to_xyxy(reg_outs)
        bboxes[..., 0::2] *= self.W
        bboxes[..., 1::2] *= self.H
        scores = cls_outs.sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    @staticmethod
    def _to_xyxy(x):
        return box_convert(x, 'cxcywh', 'xyxy')
