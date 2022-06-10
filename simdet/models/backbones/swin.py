import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import nlc_to_nchw, nchw_to_nlc, DropPath


class PatchEmbeding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            patch_size,
            stride=stride
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        h, w = x.size()[2:]
        x = nchw_to_nlc(x)
        out = self.norm(x)

        return out, h, w


class PatchMerging(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(2*embed_dim)
        self.projection = nn.Linear(2*embed_dim, embed_dim, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # (N, 4*C, H/2, W/2)
        h, w = x.size()[2:]
        x = nchw_to_nlc(x)

        x = self.norm(x)
        out = self.projection(x)

        return out, h, w


class WindowMSA(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_path_rate, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.drop_path = DropPath(drop_path_rate)
        self.n_heads = n_heads
        self.window_size = window_size

        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros(n_heads * (2*window_size-1) * (2*window_size-1))
        )
        self.register_buffer('rel_pos_index', self._get_rel_pos_index())

        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, L, C)
            x0 (torch.Tensor): (N, L, C)
            h (int): input height
            w (int): input width

        Returns:
            torch.Tensor: (N, L, C)
        """
        n = x.size(0)

        bias = self.rel_pos_bias_table.view(-1, (2*self.window_size-1)**2)[
            :, self.rel_pos_index
        ]  # (nH, window_size**2, window_size**2)

        x = self.window_partition(x, h, w, self.window_size)  # (N*nW, window_size**2, C)
        nw = x.size(0) // n
        attn_mask = bias.repeat(n * nw, 1, 1)
        x = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.window_reverse(x, h, w, self.window_size)  # (N, L, C)

        out = x0 + self.drop_path(x)

        return out

    def _get_rel_pos_index(self):
        """
        https://github.com/microsoft/Swin-Transformer/blob/cbaa0d8707db403d85ad0e13c59f2f71cd6db425/models/swin_transformer.py#L92
        """
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        rel_pos_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        return rel_pos_index

    @staticmethod
    def window_partition(x: torch.Tensor, h: int, w: int, window_size: int) -> torch.Tensor:
        """ (N, L, C) to (N*nW, window_size**2, C)

        Args:
            x (torch.Tensor): (N, L, C)
            H (int): input height
            W (int): input width
            window_size (int): window size

        Returns:
            torch.Tensor: (N*nW, window_size**2, C)
        """
        n, _, c = x.shape
        x = x.view(n, h, w, c)
        x = F.pad(x, pad=[0, 0, 0, window_size - w % window_size, 0, window_size - h % window_size])
        _, ph, pw, _ = x.shape
        x = x.view(-1, ph//window_size, window_size, pw//window_size, window_size, c)
        out = x.transpose(2, 3).contiguous().view(-1, window_size*window_size, c)

        return out

    @staticmethod
    def window_reverse(x: torch.Tensor, h: int, w: int, window_size: int) -> torch.Tensor:
        """ (N*nW, window_size**2, C) to (N, L, C)

        Args:
            x (torch.Tensor): (N*nW, window_size**2, C)
            H (int): input height
            W (int): input width
            window_size (int): window size

        Returns:
            torch.Tensor: (N, L, C)
        """
        _, _, c = x.shape
        ph, pw = h + window_size - h % window_size, w + window_size - w % window_size
        x = x.view(-1, ph//window_size, pw//window_size, window_size, window_size, c)
        x = x.transpose(2, 3).contiguous()
        out = x.view(-1, ph, pw, c)[:, :h, :w].contiguous().view(-1, h * w, c)

        return out


class ShiftedWindowMSA(WindowMSA):
    def __init__(self, embed_dim, n_heads, drop_path_rate, window_size, shift_size):
        super().__init__(embed_dim, n_heads, drop_path_rate, window_size)
        self.shift_size = shift_size
        self.mask = None

    def forward(self, x: torch.Tensor, x0: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, L, C)
            x0 (torch.Tensor): (N, L, C)
            h (int): input height
            w (int): input width

        Returns:
            torch.Tensor: (N, L, C)
        """
        n = x.size(0)

        bias = self.rel_pos_bias_table.view(-1, (2*self.window_size-1)**2)[
            :, self.rel_pos_index
        ]  # (nH, window_size**2, window_size**2)

        if self.mask is None:
            mask = torch.zeros((1, h, w, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask = mask.flatten(1, 2)
            mask = self.window_partition(mask, h, w, self.window_size)
            mask = mask - mask.transpose(1, 2)
            mask = mask.masked_fill(mask != 0, float('-inf')).masked_fill(mask == 0, float(0.0))
            self.mask = mask  # (nW, window_size**2, window_size**2)
        mask = self.mask.unsqueeze(1)

        x = self.cyclic_shift(x, h, w, -self.shift_size)
        x = self.window_partition(x, h, w, self.window_size)  # (N*nW, window_size**2, C)
        nw = x.size(0) // n
        attn_mask = bias.repeat(n * nw, 1, 1) + mask.repeat(n, self.n_heads, 1, 1).flatten(0, 1)
        x = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.window_reverse(x, h, w, self.window_size)  # (N, L, C)
        x = self.cyclic_shift(x, h, w, self.shift_size)

        out = x0 + self.drop_path(x)

        return out

    @staticmethod
    def cyclic_shift(x: torch.Tensor, h: int, w: int, shift_size: int):
        n, _, c = x.shape
        x = x.view(n, h, w, c).roll((shift_size, shift_size), dims=(1, 2))
        out = x.view(n, -1, c)

        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, drop_path_rate):
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, x0):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        out = x0 + self.drop_path(x)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, drop_path_rate, window_size, shift_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        if shift_size > 0:
            self.attn = ShiftedWindowMSA(embed_dim, n_heads, drop_path_rate, window_size, shift_size)
        else:
            self.attn = WindowMSA(embed_dim, n_heads, drop_path_rate, window_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, drop_path_rate)

    def forward(self, x, h, w):
        x = self.attn(self.norm1(x), x, h, w)
        out = self.ffn(self.norm2(x), x)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, embed_dim, n_heads, drop_path_rates, window_size, first_block=False):
        super().__init__()
        if first_block:
            self.patch = PatchEmbeding(3, embed_dim, 4, 4)
        else:
            self.patch = PatchMerging(embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                drop_path_rate=drop_path_rates[i],
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size//2
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        x, h, w = self.patch(x)
        for m in self.layers:
            x = m(x, h, w)
        out = nlc_to_nchw(x, h, w)

        return out


class SwinTransformer(nn.Module):
    def __init__(self, scale: str = 'tiny', window_size: int = 7, pretrain: str = None):
        assert scale in ('tiny', 'small', 'base')

        super().__init__()
        n_layers = self._n_layers(scale)
        embed_dim = self._embed_dim(scale)
        n_heads = self._n_heads(scale)
        dprs = torch.linspace(0, 0.3, sum(n_layers)).tolist()

        self.layers = nn.ModuleList([])
        for i in range(4):
            self.layers.append(
                TransformerBlock(
                    n_layers=n_layers[i],
                    embed_dim=embed_dim*2**i,
                    n_heads=n_heads[i],
                    drop_path_rates=dprs[sum(n_layers[:i]):sum(n_layers[:i+1])],
                    window_size=window_size,
                    first_block=i == 0
                )
            )

        for i in range(2, 6):
            setattr(self, f'C{i}', embed_dim * 2**(i-2))

        self._init_weights()
        if pretrain:
            self.load_state_dict(torch.load(pretrain))

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=pow(1.0 / fan_out, 0.5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _n_layers(self, scale: str):
        return {
            'tiny': [2, 2, 6, 2],
            'small': [2, 2, 18, 2],
            'base': [2, 2, 18, 2]
        }[scale]

    def _embed_dim(self, scale: str):
        return {
            'tiny': 96,
            'small': 96,
            'base': 128
        }[scale]

    def _n_heads(self, scale: str):
        return {
            'tiny': [3, 6, 12, 24],
            'small': [3, 6, 12, 24],
            'base': [4, 8, 16, 32]
        }[scale]
