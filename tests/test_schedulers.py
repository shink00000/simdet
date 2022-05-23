import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import os
import os.path as osp

from simdet.schedulers.multi_step_lr import MultiStepLR


def test_multi_step_lr():
    m = nn.Conv2d(3, 64, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, [50, 75], gamma=0.1, warmup_iterations=10)

    lrs = []
    for i in range(100):
        x = torch.rand(2, 3, 64, 64)
        t = torch.randint(0, 10, (2, 64, 64))
        y = m(x)
        loss = criterion(y, t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    plt.figure()
    plt.plot(lrs)
    os.makedirs(osp.join(osp.dirname(__file__), 'output'), exist_ok=True)
    plt.savefig(osp.join(osp.dirname(__file__), 'output/multi_step_lr.png'))
