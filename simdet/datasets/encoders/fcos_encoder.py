from itertools import product
import torch
import torch.nn as nn


class FCOSEncoder(nn.Module):
    INF = 1e8

    def __init__(self, input_size: list):
        super().__init__()
        H, W = input_size
        strides = [2**i for i in range(3, 8)]
        regress_ranges = [[s*4, s*8] for s in strides]
        regress_ranges[0][0], regress_ranges[-1][-1] = -1, self.INF

        all_points = []
        all_regress_ranges = []
        for stride, regress_range in zip(strides, regress_ranges):
            points = [[x, y] for y, x in product(
                range(stride//2, H, stride), range(stride//2, W, stride)
            )]
            all_points.extend(points)
            all_regress_ranges.extend([regress_range for _ in range(len(points))])

        self.all_points = torch.Tensor(all_points)
        self.all_regress_ranges = torch.Tensor(all_regress_ranges)

    def forward(self, data: tuple):
        image, bboxes, labels = data

        xs, ys = self.all_points.split(1, dim=1)
        ls, us = self.all_regress_ranges.split(1, dim=1)

        left = xs - bboxes[..., 0]
        top = ys - bboxes[..., 1]
        right = bboxes[..., 2] - xs
        bottom = bboxes[..., 3] - ys

        distances = torch.stack([left, top, right, bottom], dim=-1)
        areas = (distances[..., 0] + distances[..., 2]) * (distances[..., 1] + distances[..., 3])

        inside_bboxes = distances.min(dim=-1)[0] > 0
        inside_ranges = torch.logical_and(ls <= distances.max(dim=-1)[0], distances.max(dim=-1)[0] <= us)
        areas[inside_bboxes == 0] = self.INF
        areas[inside_ranges == 0] = self.INF
        min_area, min_area_inds = areas.min(dim=1)

        cls_targets = labels[min_area_inds]
        cls_targets[min_area == self.INF] = 0
        reg_targets = distances[range(len(min_area_inds)), min_area_inds]
        cnt_targets = torch.div(
            torch.minimum(reg_targets[..., [0, 2]], reg_targets[..., [1, 3]]),
            torch.maximum(reg_targets[..., [0, 2]], reg_targets[..., [1, 3]])
        ).prod(dim=-1, keepdim=True).sqrt()

        return (image, reg_targets, cls_targets, cnt_targets)
