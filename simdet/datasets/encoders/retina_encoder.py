from itertools import product
import torch
import torch.nn as nn
from torchvision.ops import box_iou, box_convert


class RetinaEncoder(nn.Module):
    def __init__(self, input_size: list, iou_threshs: tuple = (0.4, 0.5)):
        super().__init__()
        H, W = input_size
        strides = [2**i for i in range(3, 8)]
        prior_boxes = []
        for stride in strides:
            for cy, cx in product(range(stride//2, H, stride), range(stride//2, W, stride)):
                for aspect in [0.5, 1.0, 2.0]:
                    for scale in [0, 1/3, 2/3]:
                        h = 4 * stride * pow(2, scale) * pow(aspect, 1/2)
                        w = 4 * stride * pow(2, scale) * pow(1/aspect, 1/2)
                        prior_boxes.append([cx, cy, w, h])
        self.prior_boxes = torch.Tensor(prior_boxes)
        self.neg_thresh, self.pos_thresh = iou_threshs

    def forward(self, data: tuple):
        image, bboxes, labels = data

        ious = box_iou(bboxes, box_convert(self.prior_boxes, 'cxcywh', 'xyxy'))
        max_ious, match_ids = ious.max(dim=0)

        # force assign (all bboxes match one or more prior boxes)
        force_assign_ids = ious.argmax(dim=1)
        match_ids[force_assign_ids] = torch.arange(len(bboxes))
        max_ious[force_assign_ids] = self.pos_thresh

        offsets = self._bbox2delta(box_convert(bboxes, 'xyxy', 'cxcywh')[match_ids])
        labels = labels[match_ids]
        labels[max_ious < self.pos_thresh] = -1
        labels[max_ious < self.neg_thresh] = 0

        return (image, offsets, labels)

    def _bbox2delta(self, bboxes) -> torch.Tensor:
        dcx = (bboxes[..., 0] - self.prior_boxes[..., 0]) / self.prior_boxes[..., 2] / 0.1
        dcy = (bboxes[..., 1] - self.prior_boxes[..., 1]) / self.prior_boxes[..., 3] / 0.1
        dw = (bboxes[..., 2] / self.prior_boxes[..., 2]).log() / 0.2
        dh = (bboxes[..., 3] / self.prior_boxes[..., 3]).log() / 0.2

        return torch.stack([dcx, dcy, dw, dh], dim=-1)
