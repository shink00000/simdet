import torch
import torch.nn as nn
from torchvision.ops.boxes import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment

from .layers import nchw_to_nlc, DropPath
from .backbones import BACKBONES
from .losses import GIoULoss, FocalLoss
from .postprocesses import MultiLabelBasicProcess


class MHA(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.drop_path = DropPath(drop_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        if x0 is None:
            x0 = q
        x = self.attn(q, k, v, need_weights=False)[0]
        out = x0 + self.drop_path(x)

        return out


class FFN(nn.Module):
    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop_path = DropPath(drop_rate)

    def forward(self, x: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        if x0 is None:
            x0 = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        out = x0 + self.drop_path(x)

        return out


class DETREncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn = MHA(embed_dim, n_heads, drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        q = k = x + pe
        v = x
        x = self.attn(q, k, v, x0=x)
        x = self.norm1(x)
        x = self.ffn(x)
        out = self.norm2(x)

        return out


class DETREncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.layers = nn.ModuleList([
            DETREncoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pe)

        return x


class DETRDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_rate):
        super().__init__()
        self.attn1 = MHA(embed_dim, n_heads, drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = MHA(embed_dim, n_heads, drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, y: torch.Tensor, x: torch.Tensor, pe: torch.Tensor, object_query: torch.Tensor) -> torch.Tensor:
        q = k = y + object_query
        v = y
        y = self.attn1(q, k, v, x0=y)
        y = self.norm1(y)

        q = y + object_query
        k = x + pe
        v = x
        y = self.attn2(q, k, v, x0=y)
        y = self.norm2(y)

        y = self.ffn(y)
        out = self.norm3(y)

        return out


class DETRDecoder(nn.Module):
    def __init__(self, n_objs, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.object_query = nn.Parameter(torch.zeros(n_objs, embed_dim))
        self.layers = nn.ModuleList([
            DETRDecoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.object_query)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        outs = []
        object_queries = self.object_query.unsqueeze(0).repeat(x.size(0), 1, 1)
        y = torch.zeros_like(object_queries)
        for layer in self.layers:
            y = layer(y, x, pe, object_queries)
            outs.append(self.norm(y))
        outs = torch.stack(outs)

        return outs


class DETRHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_objs: int = 100, n_encoders: int = 6, n_decoders: int = 6):
        super().__init__()
        embed_dim = 256
        pos_dim = embed_dim // 2
        n_heads = 8
        dprs = torch.linspace(0, 0.1, n_encoders + n_decoders).tolist()

        self.projection = nn.Conv2d(in_channels, embed_dim, 1)
        self.dim_t = 10000 ** (2 * torch.div(torch.arange(pos_dim), 2, rounding_mode='trunc') / pos_dim)
        self.encoder = DETREncoder(n_encoders, embed_dim, n_heads, dprs[:n_encoders])
        self.decoder = DETRDecoder(n_objs, n_decoders, embed_dim, n_heads, dprs[n_encoders:])
        self.reg_top = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid()
        )
        self.cls_top = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def forward(self, xs: list):
        x = self.projection(xs[-1])
        if not hasattr(self, 'pe'):
            self.register_buffer('pe', self._pos_encoder(x))
        x = nchw_to_nlc(x)
        pe = nchw_to_nlc(self.pe.unsqueeze(0))
        x = self.encoder(x, pe)
        x = self.decoder(x, pe)
        reg_outs = self.reg_top(x)
        cls_outs = self.cls_top(x)

        return reg_outs, cls_outs

    def _pos_encoder(self, x: torch.Tensor, eps: float = 1e-6):
        lx, ly = x.size()[2:]
        x_embed, y_embed = torch.meshgrid(
            torch.arange(1, lx+1, dtype=torch.float32),
            torch.arange(1, ly+1, dtype=torch.float32),
            indexing='xy'
        )
        y_embed = 2 * torch.pi * y_embed / (y_embed[-1:, :] + eps)
        x_embed = 2 * torch.pi * x_embed / (x_embed[:, -1:] + eps)

        pos_x = x_embed[..., None] / self.dim_t
        pos_y = y_embed[..., None] / self.dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1).to(x.device)

        return pos

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class DETR(nn.Module):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, n_objs: int = 100,
                 lmd_iou: int = 1):
        super().__init__()

        # layers
        self.backbone = BACKBONES[backbone.pop('type')](**backbone)
        self.head = DETRHead(self.backbone.C5, n_classes, n_objs)

        # property
        self.H, self.W = input_size
        self.lmd_iou = lmd_iou

        # loss
        self.iou_loss = GIoULoss(reduction='mean')
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
            reg_out = reg_outs[i].detach().clone()
            cls_out = cls_outs[i].detach().clone()
            reg_target = reg_targets[i][:pos_targets[i]]
            cls_target = cls_targets[i][:pos_targets[i]]
            cost = -cls_out[:, cls_target-1]
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
            iou_loss = self.iou_loss(self._to_xyxy(reg_outs), self._to_xyxy(reg_targets))
            loss = cls_loss + self.lmd_iou * iou_loss
        else:
            loss = self.cls_loss(cls_outs[neg_mask], cls_targets[neg_mask])

        return loss

    def predict(self, outputs: tuple) -> tuple:
        reg_outs, cls_outs = outputs
        bboxes = self._to_xyxy(reg_outs[-1])
        bboxes[..., 0::2] *= self.W
        bboxes[..., 1::2] *= self.H
        scores = cls_outs[-1].sigmoid()
        bboxes, scores, class_ids = self.postprocess(bboxes, scores)
        return bboxes, scores, class_ids

    @staticmethod
    def _to_xyxy(x):
        return box_convert(x, 'cxcywh', 'xyxy')
