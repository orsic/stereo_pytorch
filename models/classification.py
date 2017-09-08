import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmin(nn.Module):

    def __init__(self, **config):
        super(SoftArgmin, self).__init__()
        self.max_disp = config.get('max_disp', 192)

    def forward(self, input):
        smax_out = F.softmax(-1 * input.sum(1))
        disp_mul = Variable(torch.arange(0, self.max_disp).view(1, -1, 1, 1).cuda(), requires_grad=False)
        return (smax_out * disp_mul).sum(1)