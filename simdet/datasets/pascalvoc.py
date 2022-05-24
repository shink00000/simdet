import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.io import read_image

from . import transforms as T
from .encoders import ENCODERS


class PascalVOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    N_CLASSES = 20

    def __init__(self, phase: str, data_dir: str, encoder: dict, size: list):
        self.data_list = self._get_data_list(phase, data_dir)
        self.transforms = self._get_transforms(phase, size, encoder)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple:
        image_path, annos, meta = self.data_list[idx]
        image = read_image(image_path) / 255
        bboxes = torch.tensor(annos['bboxes']).float()
        labels = torch.tensor(annos['labels']).long()
        data = (image, bboxes, labels)
        image, *targets = self.transforms(data)
        return image, targets, meta

    def _get_data_list(self, phase: str, data_dir: str):
        data_list = []
        if phase == 'test':
            phase = 'val'
        coco = COCO(f'{data_dir}/annotations/instances_{phase}.json')
        for image_id in coco.getImgIds():
            annos = {'bboxes': [], 'labels': []}
            for anno in coco.imgToAnns[image_id]:
                if anno['iscrowd'] == 1:
                    continue
                x, y, w, h = anno['bbox']
                annos['bboxes'].append([x, y, x+w, y+h])
                annos['labels'].append(anno['category_id'])
            if len(annos['bboxes']) == 0:
                continue
            info = coco.loadImgs(ids=[image_id])[0]
            image_path = osp.join(data_dir, info['file_name'])
            meta = {'image_id': image_id, 'height': info['height'], 'width': info['width']}
            data_list.append((image_path, annos, meta))
        return data_list

    def _get_transforms(self, phase, size, encoder) -> nn.Sequential:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if phase == 'train':
            transforms = nn.Sequential(
                T.PhotoMetricDistortion(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.1),
                T.Normalize(mean=mean, std=std),
                T.RandomExpand(scale_range=[1.0, 4.0], p=0.5),
                T.RandomMinIoUCrop(min_ious=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9], p=0.5),
                T.Resize(size=size),
                T.RandomHorizontalFlip(p=0.5),
                ENCODERS[encoder.pop('type')](**encoder)
            )
        elif phase == 'val':
            transforms = nn.Sequential(
                T.Normalize(mean=mean, std=std),
                T.Resize(size=size),
                ENCODERS[encoder.pop('type')](**encoder)
            )
        else:
            transforms = nn.Sequential(
                T.Normalize(mean=mean, std=std),
                T.Resize(size=size),
                ENCODERS[encoder.pop('type')](**encoder)
            )
        return transforms

    @staticmethod
    def collate_fn(batch: tuple):
        images, targets, metas = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = tuple(torch.stack(t, dim=0) for t in zip(*targets))

        return images, targets, metas
