import torch
import torch.nn as nn
from torchvision.ops.boxes import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment

from .layers import nchw_to_nlc, DropPath, SineEncoding
from .backbones import BACKBONES
from .losses import GIoULoss, FocalLoss
from .postprocesses import MultiLabelBasicProcess


class MHA(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate, kdim=None, vdim=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, kdim=kdim, vdim=vdim)
        if vdim is not None and embed_dim != vdim:
            self.proj = nn.Linear(embed_dim, vdim)
        self.drop_path = DropPath(drop_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        if x0 is None:
            x0 = q
        x = self.attn(q, k, v, need_weights=False)[0]
        if hasattr(self, 'proj'):
            x = self.proj(x)
        out = x0 + self.drop_path(x)

        return out


class FFN(nn.Module):
    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop_path = DropPath(drop_rate)

    def forward(self, x: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        if x0 is None:
            x0 = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        out = x0 + self.drop_path(x)

        return out


class DABDETREncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn = MHA(embed_dim, n_heads, drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        q = k = x + x_pe
        v = x
        x = self.attn(q, k, v, x0=x)
        x = self.norm1(x)
        x = self.ffn(x)
        out = self.norm2(x)

        return out


class DABDETREncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.layers = nn.ModuleList([
            DABDETREncoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, x_pe)

        return x


class DABDETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.self_attn = MHA(embed_dim, n_heads, drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MHA(2*embed_dim, n_heads, drop_rate, vdim=embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

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
            torch.Tensor: (N, n_objs, C) content query
        """
        # self attention
        q = k = c + pos_query
        v = c
        c = self.self_attn(q, k, v, x0=c)
        c = self.norm1(c)

        # cross attention
        q = torch.cat([c, pos_embed], dim=-1)
        k = torch.cat([x, x_pe.repeat(x.size(0), 1, 1)], dim=-1)
        v = x
        c = self.cross_attn(q, k, v, x0=c)
        c = self.norm2(c)

        # ffn
        c = self.ffn(c)
        out = self.norm3(c)

        return out


class DABDETRDecoder(nn.Module):
    def __init__(self, n_objs, n_layers, embed_dim, n_heads, drop_rates, n_classes):
        super().__init__()
        self.object_query = nn.Parameter(torch.zeros(n_objs, 4))
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            DABDETRDecoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])
        self.pos_encoding = SineEncoding(embed_dim//2, temperature=20)
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
        for layer in self.layers:
            anchors.append(object_query)

            # positional query for self attention
            pe = self.pos_encoding(object_query.sigmoid())
            pos_query = self.mlp_pe_proj(pe)

            # positional (modulated) embedding for cross attention
            pos_embed = pe[..., :self.embed_dim] * self.mlp_ref_xy(c)
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
        self.pos_encoding = SineEncoding(embed_dim//2, temperature=20)
        self.x_pe = None
        self.encoder = DABDETREncoder(n_encoders, embed_dim, n_heads, dprs[:n_encoders])
        self.decoder = DABDETRDecoder(n_objs, n_decoders, embed_dim, n_heads, dprs[n_encoders:], n_classes)

        self._init_weights()

    def forward(self, xs: list):
        x = self.projection(xs[-1])
        _, _, h, w = x.shape
        x = nchw_to_nlc(x)
        if self.x_pe is None:
            pos_x = torch.arange(w, device=x.device).view(1, -1).repeat(h, 1).flatten() / w
            pos_y = torch.arange(h, device=x.device).view(-1, 1).repeat(1, w).flatten() / h
            pos = torch.stack([pos_x, pos_y], dim=-1).unsqueeze(0)
            self.x_pe = self.pos_encoding(pos)
        x = self.encoder(x, self.x_pe)
        reg_outs, cls_outs = self.decoder(x, self.x_pe)

        return reg_outs, cls_outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class DABDETR(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, n_objs: int = 100,
                 lmd_l1: int = 1, lmd_iou: int = 1):
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
