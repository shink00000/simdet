import torch
import torch.nn as nn
import numpy as np
from torchvision.ops.boxes import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment

from .layers import nchw_to_nlc, MultiheadAttentionV2, FeedForwardNetwork, DropPath, SineEncoding
from .backbones import BACKBONES
from .losses import GIoULoss, FocalLoss
from .postprocesses import MultiLabelBasicProcess


class DABDETREncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn = MultiheadAttentionV2(embed_dim, n_heads, mode='add')
        self.drop1 = DropPath(drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, activation=nn.PReLU)
        self.drop2 = DropPath(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        shortcut = x
        q_c, q_p = k_c, k_p = x, x_pe
        v = x
        x = self.attn(q_c, q_p, k_c, k_p, v)
        x = self.norm1(x + self.drop1(shortcut))

        shortcut = x
        x = self.ffn(x)
        out = self.norm2(x + self.drop2(shortcut))

        return out


class DABDETREncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.layers = nn.ModuleList([
            DABDETREncoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, x_pe * self.mlp(x))

        return x


class DABDETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate, is_first):
        super().__init__()
        self.self_attn = MultiheadAttentionV2(embed_dim, n_heads, mode='add')
        self.drop1 = DropPath(drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadAttentionV2(embed_dim, n_heads, mode='cat')
        self.drop2 = DropPath(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, activation=nn.PReLU)
        self.drop3 = DropPath(drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.is_first = is_first

    def forward(self, c: torch.Tensor, x: torch.Tensor, x_pe: torch.Tensor,
                pos_query: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c (torch.Tensor): (N, n_objs, C)
            x (torch.Tensor): (N, L, C)
            x_pe (torch.Tensor): (1, L, C)
            pos_query (torch.Tensor): (N, n_objs, C) for self attention
            pos_embed (torch.Tensor): (N, n_objs, C) for cross attention

        Returns:
            torch.Tensor: (N, n_objs, C)
        """
        # self attention
        shortcut = c
        q_c, q_p = k_c, k_p = c, pos_query
        v = c
        c = self.self_attn(q_c, q_p, k_c, k_p, v)
        c = self.norm1(c + self.drop1(shortcut))

        # cross attention
        shortcut = c
        if self.is_first:
            q_c, q_p = c, pos_embed
            k_c, k_p = x, x_pe
        else:
            q_c, q_p = c + pos_query, pos_embed
            k_c, k_p = x + x_pe, x_pe
        v = x
        c = self.cross_attn(q_c, q_p, k_c, k_p, v)
        c = self.norm2(c + self.drop2(shortcut))

        # ffn
        shortcut = c
        c = self.ffn(c)
        out = self.norm3(c + self.drop3(shortcut))

        return out


class DABDETRDecoder(nn.Module):
    def __init__(self, n_objs, n_layers, embed_dim, n_heads, drop_rates, n_classes):
        super().__init__()
        self.object_query = nn.Parameter(torch.zeros(n_objs, 4))
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            DABDETRDecoderLayer(embed_dim, n_heads, drop_rates[i], is_first=i == 0)
            for i in range(n_layers)
        ])
        self.pos_encoding = SineEncoding(2*embed_dim, 4, temperature=20)
        self.mlp_pe_proj = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_ref_xy = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_ref_wh = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2)
        )
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
        outs, anchors = [], []
        object_query = self.object_query.unsqueeze(0).repeat(x.size(0), 1, 1)
        c = object_query.new_zeros((*object_query.shape[:2], x.shape[-1]))
        for i, layer in enumerate(self.layers):
            anchors.append(object_query)

            # positional query for self attention
            pe = self.pos_encoding(object_query.sigmoid())
            pos_query = self.mlp_pe_proj(pe)

            # positional (modulated) embedding for cross attention
            pos_embed = pe[..., :self.embed_dim] * (1 if i == 0 else self.mlp_ref_xy(c))
            w_ref, h_ref = self.mlp_ref_wh(c).sigmoid().split(1, dim=-1)
            w, h = object_query[..., 2:].sigmoid().split(1, dim=-1)
            pos_embed[..., :self.embed_dim//2] *= w_ref / w
            pos_embed[..., self.embed_dim//2:] *= h_ref / h

            c = layer(c, x, x_pe, pos_query, pos_embed)
            outs.append(self.norm(c))
            object_query = (object_query + self.reg_top(c)).detach()

        outs = torch.stack(outs)
        anchors = torch.stack(anchors)
        reg_outs = (self.reg_top(outs) + anchors).sigmoid()
        cls_outs = self.cls_top(outs)

        return reg_outs, cls_outs


class DABDETRHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_objs: int = 100, n_encoders: int = 6, n_decoders: int = 6):
        super().__init__()
        embed_dim = 256
        n_heads = 8
        dprs = torch.linspace(0, 0.1, n_encoders + n_decoders).tolist()

        self.projection = nn.Conv2d(in_channels, embed_dim, 1)
        self.pos_encoding = SineEncoding(embed_dim, 2, temperature=20)
        self.x_pe = None
        self.encoder = DABDETREncoder(n_encoders, embed_dim, n_heads, dprs[:n_encoders])
        self.decoder = DABDETRDecoder(n_objs, n_decoders, embed_dim, n_heads, dprs[n_encoders:], n_classes)

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
        nn.init.constant_(self.decoder.cls_top.bias, np.log((1 - 0.01) / 0.01))


class DABDETR(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, n_objs: int = 100,
                 lmd_l1: int = 5, lmd_iou: int = 2):
        super().__init__()

        # layers
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.head = DABDETRHead(self.backbone.C5, n_classes, n_objs)

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
