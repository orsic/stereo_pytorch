import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class L1LossSparse(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(L1LossSparse, self).__init__(size_average=False)

    def forward(self, output, target):
        N = (target > 0).float().sum()
        if (N.data > 0).all():
            loss = super(L1LossSparse, self).forward(output * (target > 0).float(), target)
            loss /= N.float()
        else:
            loss = 0
        return loss


class L1LossDense(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(L1LossDense, self).__init__(size_average=True)


class L1Loss(nn.L1Loss):
    pass


class ConsistencyLoss(nn.Module):
    def forward(self, dispL, dispLrec, dispR, explain):
        explain_target = Variable(torch.ones(torch.numel(dispL.data)).view(explain.size()).cuda(async=True),
                                  requires_grad=False)
        explain_loss = F.binary_cross_entropy_with_logits(explain, explain_target, size_average=False)
        return torch.mean((torch.abs(dispL - dispLrec)) * explain_loss)

class L1SparseConsistencyLoss(nn.Module):
    def __init__(self, **config):
        super(L1SparseConsistencyLoss, self).__init__()
        self.supervised_loss = L1LossSparse()
        self.consistency_loss = ConsistencyLoss()
        self.max_disp = config.get('max_disp', 192)
        self.consistency_lambda = config.get('consistency_lambda', 1.0)

    def forward(self, target, dispL, dispLrec, dispR, explain):
        supervised_loss = self.supervised_loss(dispL, target)
        consistency_loss = self.consistency_lambda * self.consistency_loss(dispL, dispLrec, dispR,
                                                                           explain) / self.max_disp
        return supervised_loss + consistency_loss
