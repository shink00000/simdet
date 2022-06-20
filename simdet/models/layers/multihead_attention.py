import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dims = embed_dim
        self.n_heads = n_heads
        self.d = (embed_dim//n_heads) ** 0.5
        self.in_proj_q = nn.Linear(embed_dim, embed_dim)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): (N, L, C)
            k (torch.Tensor): (N, S, C)
            v (torch.Tensor): (N, S, C)
            attn_mask (torch.Tensor, optional): (N, H, L, S). Defaults to None.

        Returns:
            torch.Tensor: (N, L, C)
        """
        q = self.in_proj_q(q)
        k = self.in_proj_k(k)
        v = self.in_proj_v(v)

        q = self._split_head(q, self.n_heads)
        k = self._split_head(k, self.n_heads)
        v = self._split_head(v, self.n_heads)

        attn = torch.matmul(q / self.d, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self._combine_head(out)
        out = self.out_proj(out)

        return out

    @staticmethod
    def _split_head(x, h):
        n, l, c = x.shape
        return x.reshape(n, l, h, c//h).transpose(1, 2)

    @staticmethod
    def _combine_head(x):
        return x.transpose(1, 2).flatten(2)


class MultiheadAttentionV2(MultiheadAttention):
    def __init__(self, embed_dim, n_heads, mode='cat'):
        assert mode in ('add', 'cat')
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dim
        self.n_heads = n_heads
        self.d = (embed_dim//n_heads) ** 0.5
        self.mode = mode
        self.in_proj_q_c = nn.Linear(embed_dim, embed_dim)
        self.in_proj_q_p = nn.Linear(embed_dim, embed_dim)
        self.in_proj_k_c = nn.Linear(embed_dim, embed_dim)
        self.in_proj_k_p = nn.Linear(embed_dim, embed_dim)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_c: torch.Tensor, q_p: torch.Tensor, k_c: torch.Tensor, k_p: torch.Tensor,
                v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q_c (torch.Tensor): (N, L, C)
            q_p (torch.Tensor): (N, L, C)
            k_c (torch.Tensor): (N, S, C)
            k_p (torch.Tensor): (N, S, C)
            v (torch.Tensor): (N, S, C)
            attn_mask (torch.Tensor, optional): (N, H, L, S). Defaults to None.

        Returns:
            torch.Tensor: (N, L, C)
        """
        if self.mode == 'cat':
            q = torch.cat([self.in_proj_q_c(q_c), self.in_proj_q_p(q_p)], dim=-1)
            k = torch.cat([self.in_proj_k_c(k_c), self.in_proj_k_p(k_p)], dim=-1)
        elif self.mode == 'add':
            q = self.in_proj_q_c(q_c) + self.in_proj_q_p(q_p)
            k = self.in_proj_k_c(k_c) + self.in_proj_k_p(k_p)
        v = self.in_proj_v(v)

        q = self._split_head(q, self.n_heads)
        k = self._split_head(k, self.n_heads)
        v = self._split_head(v, self.n_heads)

        attn = torch.matmul(q / self.d, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self._combine_head(out)
        out = self.out_proj(out)

        return out
