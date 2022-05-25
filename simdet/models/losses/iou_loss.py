import torch
import torch.nn as nn
from torchvision.ops import box_iou


class IoULoss(nn.Module):
    """ IoU Loss

    Args:
        input (torch.Tensor): [N, 4] (x, y, x, y)
        target (torch.Tensor): [N, 4] (x, y, x, y)

    Examples:
        >>> iou_loss = IouLoss(reduction='mean')
        >>> input = torch.rand(N, 4)
        >>> target = torch.rand(N, 4)
        >>> loss = lou_loss(input, target)

    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        iou = box_iou(input, target).diagonal().clamp(min=1e-6)
        loss = -iou.log()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class IoULossWithDistance(nn.Module):
    """ IoU Loss (case: input and target are distance tensors)

    Args:
        input (torch.Tensor): [N, 4]
        target (torch.Tensor): [N, 4]

    Examples:
        >>> iou_loss = IouLoss(reduction='mean')
        >>> input = torch.rand(N, 4)
        >>> target = torch.rand(N, 4)
        >>> loss = lou_loss(input, target)

    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        inter = self._calc_area(torch.minimum(input, target))
        union = self._calc_area(input) + self._calc_area(target) - inter
        iou = inter / union
        loss = -iou.log()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _calc_area(self, t):
        return (t[..., 0] + t[..., 2]) * (t[..., 1] + t[..., 3])
