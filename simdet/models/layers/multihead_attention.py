import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """
    A little modified: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self, embed_dims: int, n_heads: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dims, 3*embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.d = (embed_dims//n_heads) ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                b: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): (N, L, C)
            k (torch.Tensor): (N, S, C)
            v (torch.Tensor): (N, S, C)
            b (torch.Tensor): (1, H, L, L)
            mask (torch.Tensor): (M, L, L)

        Returns:
            torch.Tensor: (N, L, C)
        """
        in_proj_w_q, in_proj_w_k, in_proj_w_v = self.in_proj.weight.chunk(3)
        in_proj_b_q, in_proj_b_k, in_proj_b_v = self.in_proj.bias.chunk(3)
        q = self._nlc_to_nhlc(torch.matmul(q, in_proj_w_q.T) + in_proj_b_q)
        k = self._nlc_to_nhlc(torch.matmul(k, in_proj_w_k.T) + in_proj_b_k)
        v = self._nlc_to_nhlc(torch.matmul(v, in_proj_w_v.T) + in_proj_b_v)

        attn = torch.matmul(q / self.d, k.transpose(-2, -1))  # (N, nH, L, L)
        if b is not None:
            attn = attn + b
        if mask is not None:
            n, nh, l, _ = attn.shape
            nw = mask.shape[0]
            attn = attn.view(n//nw, nw, nh, l, l)
            mask = mask.view(1, nw, 1, l, l)
            attn = attn + mask
            attn = attn.view(n, nh, l, l)
        attn = F.softmax(attn, dim=-1)  # (N, nH, L, L)
        v = torch.matmul(attn, v)  # (N, nH, L, C')

        v = v.transpose(1, 2).flatten(2)  # (N, L, C)
        out = self.out_proj(v)  # (N, L, C)

        return out

    def _nlc_to_nhlc(self, x):
        N, L, C = x.shape
        H = self.n_heads
        return x.reshape(N, L, H, C//H).transpose(1, 2)


if __name__ == '__main__':
    m1 = nn.MultiheadAttention(32, 2, batch_first=True)
    m2 = MultiheadAttention(32, 2)
    m2.in_proj.weight = m1.in_proj_weight
    m2.in_proj.bias = m1.in_proj_bias
    m2.out_proj.weight = m1.out_proj.weight
    m2.out_proj.bias = m1.out_proj.bias

    q = torch.rand(2, 12, 32)
    k = torch.rand(2, 6, 32)
    v = torch.rand(2, 6, 32)

    y1, _ = m1(q, k, v)
    y2 = m2(q, k, v)
    print((y1 == y2).float().mean())
