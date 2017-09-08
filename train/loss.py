import torch.nn as nn


class L1LossSparse(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(L1LossSparse, self).__init__(size_average=False)

    def forward(self, input, target):
        loss = super(L1LossSparse, self).forward(input, target)
        N = (target > 0).sum()
        if (N.data > 0).all():
            loss /= N.float()
        else:
            loss = 0
        return loss


class L1LossDense(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(L1LossDense, self).__init__(size_average=True)
