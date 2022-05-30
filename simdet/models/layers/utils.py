import torch


def nlc_to_nchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    n, _, c = x.shape
    return x.transpose(1, 2).view(n, c, h, w)


def nchw_to_nlc(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)
