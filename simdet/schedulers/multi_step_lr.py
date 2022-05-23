from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, warmup_iterations=0,
                 warmup_ratio=0.1, last_epoch=-1, verbose=False):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iterations = warmup_iterations
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_iterations:
            factor = (1 - self.warmup_ratio) * self.last_epoch / self.warmup_iterations + self.warmup_ratio
            return [group['initial_lr'] * factor for group in self.optimizer.param_groups]
        else:
            gamma = self.gamma if self.last_epoch in self.milestones else 1
            return [group['lr'] * gamma for group in self.optimizer.param_groups]
