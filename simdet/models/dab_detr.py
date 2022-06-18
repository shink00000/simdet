
import torch
import torch.nn as nn

from .layers import nchw_to_nlc
from .backbones import BACKBONES
from .losses import GIoULoss, FocalLoss
from .postprocesses import MultiLabelBasicProcess
from .detr import MHA, FFN, SineEncoding, DETREncoder, DETR


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
                anchor_query: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c (torch.Tensor): (N, n_objs, C)
            x (torch.Tensor): (N, L, C)
            x_pe (torch.Tensor): (1, L, C)
            anchor_query (torch.Tensor): (N, n_objs, C) for self attention
            pos_embed (torch.Tensor): (N, n_objs, C) for cross attention

        Returns:
            torch.Tensor: (N, n_objs, C) content query
        """
        # self attention
        q = k = c + anchor_query
        v = c
        c = self.self_attn(q, k, v, x0=c)
        c = self.norm1(c)

        # cross attention
        q = torch.cat([c, pos_embed], dim=-1)
        k = torch.cat([x, x_pe], dim=-1)
        v = x
        c = self.cross_attn(q, k, v, x0=c)
        c = self.norm2(c)

        # ffn
        c = self.ffn(c)
        out = self.norm3(c)

        return out


class DABDETRDecoder(nn.Module):
    def __init__(self, n_objs, n_layers, embed_dim, n_heads, drop_rates):
        super().__init__()
        self.embed_dim = embed_dim
        self.anchor_query = nn.Parameter(torch.zeros(n_objs, 4))
        self.layers = nn.ModuleList([
            DABDETRDecoderLayer(embed_dim, n_heads, drop_rates[i])
            for i in range(n_layers)
        ])
        self.pos_encoding = SineEncoding(2*embed_dim, 4, temperature=20)
        self.mlp_proj = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_cond_scale = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_ref_wh = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2)
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4)
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.anchor_query)

    def forward(self, x: torch.Tensor, x_pe: torch.Tensor) -> torch.Tensor:
        outs, anchors = [], []
        anchor_query = self.anchor_query.unsqueeze(0).repeat(x.size(0), 1, 1)
        c = torch.zeros((*anchor_query.shape[:2], x.shape[-1]))
        for layer in self.layers:
            anchors.append(anchor_query)

            # positional query for self attention
            pe = self.pos_encoding(anchor_query.sigmoid())
            pos_query = self.mlp_proj(pe)

            # positional (modulated) embedding for cross attention
            pos_embed = pe[..., :self.embed_dim] * self.mlp_cond_scale(c)
            w_ref, h_ref = self.mlp_ref_wh(c).sigmoid().split(1, dim=-1)
            w, h = anchor_query[..., 2:].sigmoid().split(1, dim=-1)
            pos_embed[..., :self.embed_dim//2] *= w_ref / w
            pos_embed[..., self.embed_dim//2:] *= h_ref / h

            c = layer(c, x, x_pe, pos_query, pos_embed)
            outs.append(self.norm(c))
            anchor_query = (anchor_query + self.mlp_offset(c)).detach()

        outs = torch.stack(outs)
        anchors = torch.stack(anchors)

        return outs, anchors


class DABDETRHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_objs: int = 100, n_encoders: int = 6, n_decoders: int = 6):
        super().__init__()
        embed_dim = 256
        n_heads = 8
        dprs = torch.linspace(0, 0.1, n_encoders + n_decoders).tolist()

        self.projection = nn.Conv2d(in_channels, embed_dim, 1)
        self.pos_encoding = SineEncoding(embed_dim, 2)
        self.x_pe = None
        self.encoder = DETREncoder(n_encoders, embed_dim, n_heads, dprs[:n_encoders])
        self.decoder = DABDETRDecoder(n_objs, n_decoders, embed_dim, n_heads, dprs[n_encoders:])
        self.reg_top = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4)
        )
        self.cls_top = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def forward(self, xs: list):
        x = self.projection(xs[-1])
        n, _, h, w = x.shape
        x = nchw_to_nlc(x)
        if self.x_pe is None:
            pos_x = torch.arange(w, device=x.device).view(1, -1).repeat(h, 1).flatten() / w
            pos_y = torch.arange(h, device=x.device).view(-1, 1).repeat(1, w).flatten() / h
            pos = torch.stack([pos_x, pos_y], dim=-1).unsqueeze(0).repeat(n, 1, 1)
            self.x_pe = self.pos_encoding(pos)
        x = self.encoder(x, self.x_pe)
        x, anchors = self.decoder(x, self.x_pe)
        reg_outs = (self.reg_top(x) + anchors).sigmoid()
        cls_outs = self.cls_top(x)

        return reg_outs, cls_outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class DABDETR(DETR):
    def __init__(self, backbone: dict, n_classes: int, input_size: list, n_objs: int = 100,
                 lmd_l1: int = 1, lmd_iou: int = 1):
        super(DETR, self).__init__()

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
