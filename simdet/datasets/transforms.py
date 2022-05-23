from random import randint, uniform, random, choice
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional as F


class PhotoMetricDistortion(nn.Module):
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.1):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, data: tuple):
        image, bboxes, labels = data
        mode = randint(0, 1)
        if mode == 0:
            image = self._adjust_contrast(image)
        image = self._adjust_brightness(image)
        image = self._rgb_to_hsv(image)
        image = self._adjust_saturation(image)
        image = self._adjust_hue(image)
        image = self._hsv_to_rgb(image)
        if mode == 1:
            image = self._adjust_contrast(image)
        return (image, bboxes, labels)

    def _adjust_brightness(self, image):
        if randint(0, 1):
            image = image + uniform(-self.brightness, self.brightness)
            image = image.clip(0, 1)
        return image

    def _adjust_contrast(self, image):
        if randint(0, 1):
            image = image * uniform(1-self.contrast, 1+self.contrast)
            image = image.clip(0, 1)
        return image

    def _adjust_saturation(self, image):
        if randint(0, 1):
            image[1] = image[1] * uniform(1-self.saturation, 1+self.saturation)
            image[1] = image[1].clip(0, 1)
        return image

    def _adjust_hue(self, image):
        if randint(0, 1):
            image[0] = image[0] + 180 * uniform(-self.hue, self.hue)
            image[0] = image[0] % 360
        return image

    def _rgb_to_hsv(self, image, eps=1e-8):
        # https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        r, g, b = image
        max_rgb, argmax_rgb = image.max(0)
        min_rgb, _ = image.min(0)

        v = max_rgb
        s = torch.where(v != 0, (v - min_rgb) / v, torch.zeros_like(v))
        h = torch.stack([
            60 * (g - b) / (v - min_rgb + eps),
            60 * (b - r) / (v - min_rgb + eps) + 120,
            60 * (r - g) / (v - min_rgb + eps) + 240
        ], dim=0).gather(dim=0, index=argmax_rgb[None]).squeeze(0) % 360

        return torch.stack([h, s, v], dim=0)

    def _hsv_to_rgb(self, image):
        # https://www.rapidtables.com/convert/color/hsv-to-rgb.html
        h, s, v = image
        c = v * s
        x = c * (1 - (h / 60 % 2 - 1).abs())
        m = v - c
        z = torch.zeros_like(c)
        h_id = (h / 60).long().clip(0, 5)
        r_ = torch.stack([c, x, z, z, x, c], dim=0).gather(dim=0, index=h_id[None])
        g_ = torch.stack([x, c, c, x, z, z], dim=0).gather(dim=0, index=h_id[None])
        b_ = torch.stack([z, z, x, c, c, x], dim=0).gather(dim=0, index=h_id[None])

        return torch.cat([r_ + m, g_ + m, b_ + m], dim=0)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.normalize = T.Normalize(mean, std)

    def forward(self, data: tuple):
        image, bboxes, labels = data
        image = self.normalize(image)

        return (image, bboxes, labels)


class RandomExpand(nn.Module):
    def __init__(self, scale_range: list = [1.0, 4.0], p: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, data: tuple):
        image, bboxes, labels = data

        if random() < self.p:
            bboxes = bboxes.clone()
            h, w = image.shape[1:]
            scale = uniform(*self.scale_range)
            sh, sw = int(h * scale), int(w * scale)
            off_h_t = randint(0, sh - h)
            off_h_b = sh - h - off_h_t
            off_w_l = randint(0, sw - w)
            off_w_r = sw - w - off_w_l
            pad_lengths = [off_w_l, off_h_t, off_w_r, off_h_b]
            image = F.pad(image, pad_lengths)
            bboxes[:, [0, 2]] += pad_lengths[0]
            bboxes[:, [1, 3]] += pad_lengths[1]

        return (image, bboxes, labels)


class RandomMinIoUCrop(nn.Module):
    def __init__(self, min_ious: list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                 n_cands: int = 50, aspect_range: list = [0.5, 2.0], p: float = 0.5):
        super().__init__()
        self.min_ious = min_ious
        self.n_cands = n_cands
        self.aspect_range = aspect_range
        self.p = p

    def forward(self, data: tuple):
        image, bboxes, labels = data

        if random() < self.p:
            iou_thresh = choice(self.min_ious)

            # create candidate crop regions
            h, w = image.shape[1:]
            wh = torch.Tensor([w, h])
            whcrop = (wh * torch.empty(self.n_cands, 2).uniform_(0.3, 1)).int()
            xymin = ((wh - whcrop) * torch.rand(self.n_cands, 2)).int()
            xymax = xymin + whcrop
            crop_regions = torch.cat([xymin, xymax], dim=1)

            # filter by conditions
            aspect_ratio = whcrop[:, 0] / whcrop[:, 1]
            crop_regions = crop_regions[aspect_ratio.clip(*self.aspect_range) == aspect_ratio]
            min_ious = self._calc_iou(crop_regions, bboxes).min(dim=1)[0]
            crop_regions = crop_regions[min_ious > iou_thresh]
            if len(crop_regions) > 0:
                l, t, r, b = crop_regions[0]
                image = image[:, t:b, l:r]
                bboxes = bboxes.clone()
                bboxes[:, [0, 2]] -= l
                bboxes[:, [1, 3]] -= t
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(min=0, max=r-l)
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(min=0, max=b-t)

        return (image, bboxes, labels)

    def _calc_iou(self, box1: torch.Tensor, box2: torch.Tensor):
        inter = (
            torch.minimum(box1[:, None, 2:], box2[:, 2:]) - torch.maximum(box1[:, None, :2], box2[:, :2])
        ).clip(0).prod(dim=-1)
        area = (box2[:, 2:] - box2[:, :2]).prod(dim=-1)
        return inter / area


class Resize(nn.Module):
    def __init__(self, size: list):
        super().__init__()
        self.size = size

    def forward(self, data: tuple) -> tuple:
        image, bboxes, labels = data
        h, w = image.shape[1:]
        new_h, new_w = self.size
        image = F.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        bboxes[:, [0, 2]] *= (new_w / w)
        bboxes[:, [1, 3]] *= (new_h / h)

        return (image, bboxes, labels)


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, data: tuple) -> tuple:
        image, bboxes, labels = data
        if random() < self.p:
            image = F.hflip(image)
            bboxes[:, [0, 2]] = image.shape[2] - bboxes[:, [2, 0]]

        return (image, bboxes, labels)
