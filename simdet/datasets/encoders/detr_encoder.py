import torch
import torch.nn as nn
from torchvision.ops import box_convert


class DETREncoder(nn.Module):
    def __init__(self, n_rows: int = 50):
        super().__init__()
        self.n_rows = n_rows

    def forward(self, data: tuple):
        image, bboxes, labels = data
        _, h, w = image.shape
        bboxes[:, 0::2] /= w
        bboxes[:, 1::2] /= h

        reg_targets = torch.zeros((self.n_rows, 4))
        cls_targets = torch.zeros((self.n_rows,), dtype=torch.long)

        pos_targets = torch.tensor(len(bboxes))
        reg_targets[:pos_targets] = box_convert(bboxes, 'xyxy', 'cxcywh')
        cls_targets[:pos_targets] = labels

        return (image, reg_targets, cls_targets, pos_targets)
