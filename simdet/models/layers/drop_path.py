import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_path_rate):
        super().__init__()
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        if self.drop_path_rate == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)

        return x * random_tensor

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_path_rate}'
