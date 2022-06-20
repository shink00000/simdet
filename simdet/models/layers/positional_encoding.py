import torch
import torch.nn as nn


class SineEncoding(nn.Module):
    def __init__(self, pos_dim: int, temperature: int = 10000):
        super().__init__()
        self.temperature = temperature
        dim_t = temperature ** (2 * torch.div(torch.arange(pos_dim), 2, rounding_mode='trunc') / pos_dim)
        self.register_buffer('dim_t', dim_t)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (torch.Tensor): (N, L, n_dim)

        Returns:
            torch.Tensor: (N, L, n_dim)
        """
        pe = []
        for i in range(pos.size(-1)):
            p = 2 * torch.pi * pos[..., i, None] / self.dim_t
            p = torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2)
            pe.append(p)
        pe = torch.cat(pe, dim=-1)

        return pe

    def extra_repr(self) -> str:
        return f'temperature={self.temperature}'
