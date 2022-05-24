import yaml
import os.path as osp
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

from ..datasets import DATASETS
from ..models import MODELS
from ..optimizers import OPTIMIZERS
from ..schedulers import SCHEDULERS
from ..metrics.mean_ap import BBoxMeanAP


class Config:
    def __init__(self, config_path: str, resume_from: str = None):
        self.cfg = self._load_config(config_path)
        if resume_from and osp.exists(resume_from):
            self.checkpoint = self._load_checkpoint(resume_from)
            self.start_epoch = self.checkpoint['epoch'] + 1
        else:
            self.start_epoch = 1
        self.epochs = self.cfg['runtime']['epochs']

    def build_dataloader(self, phase: str) -> DataLoader:
        cfg = deepcopy(self.cfg['dataset'])
        loader_cfg = cfg.pop('loader', {})
        if phase == 'test':
            loader_cfg.update({'batch_size': 1})
        dataset = DATASETS[cfg.pop('type')](phase=phase, **cfg)
        self.n_classes = dataset.N_CLASSES
        self.class_names = dataset.CLASS_NAMES
        dataloader = DataLoader(
            dataset,
            shuffle=phase == 'train',
            drop_last=phase == 'train',
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **loader_cfg
        )
        return dataloader

    def build_model(self) -> nn.Module:
        cfg = deepcopy(self.cfg['model'])
        model = MODELS[cfg.pop('type')](**{**cfg, 'n_classes': self.n_classes})
        if hasattr(self, 'checkpoint'):
            model.load_state_dict(self.checkpoint['model'])
        return model

    def build_optimizer(self, model) -> nn.Module:
        cfg = deepcopy(self.cfg['optimizer'])
        optimizer = OPTIMIZERS[cfg.pop('type')](model.parameters(cfg), **cfg)
        if hasattr(self, 'checkpoint'):
            optimizer.load_state_dict(self.checkpoint['optimizer'])
        return optimizer

    def build_scheduler(self, optimizer) -> nn.Module:
        cfg = deepcopy(self.cfg['scheduler'])
        scheduler = SCHEDULERS[cfg.pop('type')](optimizer, **cfg)
        if hasattr(self, 'checkpoint'):
            scheduler.load_state_dict(self.checkpoint['scheduler'])
        return scheduler

    def build_metric(self) -> nn.Module:
        metric = BBoxMeanAP(**self.cfg['metric'])
        return metric

    @staticmethod
    def _load_config(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg

    @staticmethod
    def _load_checkpoint(resume_from):
        checkpoint = torch.load(resume_from, map_location='cpu')
        return checkpoint
