import torch
import torch.nn as nn


class SineEncoding(nn.Module):
    def __init__(self, embed_dim: int, n_dim: int, temperature: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_dim = n_dim
        self.temperature = temperature
        pos_dim = embed_dim // n_dim
        dim_t = temperature ** (2 * torch.div(torch.arange(pos_dim), 2, rounding_mode='trunc') / pos_dim)
        self.register_buffer('dim_t', dim_t)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (torch.Tensor): (N, L, n_dim)

        Returns:
            torch.Tensor: (N, L, n_dim)
        """
        assert self.n_dim == pos.size(-1)
        pe = []
        for i in range(self.n_dim):
            p = 2 * torch.pi * pos[..., i, None] / self.dim_t
            p = torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2)
            pe.append(p)
        pe = torch.cat(pe, dim=-1)

        return pe

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, temperature={self.temperature}'
