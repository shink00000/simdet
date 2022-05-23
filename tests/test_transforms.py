import os
import os.path as osp

import numpy as np
import torch
from torchvision.io import read_image
from PIL import Image, ImageDraw

from simdet.datasets import transforms as T
from simdet.datasets import PascalVOCDataset


def load():
    dir_path = osp.join(osp.dirname(__file__), 'test_data')
    image = read_image(osp.join(dir_path, 'image.jpg')) / 255
    bboxes = torch.from_numpy(np.load(osp.join(dir_path, 'bboxes.npy')))
    labels = torch.from_numpy(np.load(osp.join(dir_path, 'labels.npy')))
    data = (image, bboxes,  labels)
    return data


def save(image, bboxes, labels, file_name):
    dir_path = osp.join(osp.dirname(__file__), 'output')
    os.makedirs(dir_path, exist_ok=True)
    image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype('uint8'))
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        draw.rectangle(bbox.numpy().astype(int).tolist(), outline='white', width=3)
        draw.text(bbox.numpy().astype(int).tolist()[:2], PascalVOCDataset.CLASS_NAMES[int(label)-1], fill='white')

    image.save(osp.join(dir_path, f'{file_name}.png'))


def test_resize():
    data = load()
    t = T.Resize([256, 512])
    image, bboxes, labels = t(data)
    save(image, bboxes, labels, 'resize')
