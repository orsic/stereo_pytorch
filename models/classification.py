import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .sampler import reconstruct
from .unary import SeLuResblock


class SoftArgmin(nn.Module):
    def __init__(self, **config):
        super(SoftArgmin, self).__init__()
        self.max_disp = config.get('max_disp', 192)
        self._disp_mul = None

    def forward(self, input):
        smax_out = F.softmax(-1 * input.sum(1))
        return (smax_out * self.disp_mul).sum(1)

    @property
    def disp_mul(self):
        if self._disp_mul is None:
            self._disp_mul = Variable(torch.arange(0, self.max_disp).view(1, -1, 1, 1).cuda(), requires_grad=False)
        return self._disp_mul


class SoftArgmax(nn.Module):
    def __init__(self, **config):
        super(SoftArgmax, self).__init__()
        self.max_disp = config.get('max_disp', 192)
        self._disp_mul = None

    def forward(self, input):
        smax_out = F.softmax(input.sum(1))
        return (smax_out * self.disp_mul).sum(1)

    @property
    def disp_mul(self):
        if self._disp_mul is None:
            self._disp_mul = Variable(torch.arange(0, self.max_disp).view(1, -1, 1, 1).cuda(), requires_grad=False)
        return self._disp_mul


class SoftArgminOclussion(nn.Module):
    def __init__(self, **config):
        super(SoftArgminOclussion, self).__init__()
        self.max_disp = config.get('max_disp', 192)
        self._disp_mul = None
        self.explain = nn.Sequential(
            SeLuResblock(4, 64, 3),
            SeLuResblock(64, 64, 3),
            SeLuResblock(64, 64, 3),
            SeLuResblock(64, 64, 3),
            SeLuResblock(64, 4, 3),
            nn.Conv2d(4, 1, 1, bias=False)
        )

    @property
    def disp_mul(self):
        if self._disp_mul is None:
            self._disp_mul = Variable(torch.arange(0, self.max_disp).view(1, -1, 1, 1).cuda(async=True),
                                      requires_grad=False)
        return self._disp_mul

    def forward(self, inputL, inputR):
        smax_out_L = F.softmax(-1 * inputL.sum(1))
        dispL = (smax_out_L * self.disp_mul).sum(1)

        smax_out_R = F.softmax(-1 * inputR.sum(1))
        dispR = (smax_out_R * self.disp_mul).sum(1)

        dispLrec = reconstruct(dispR, dispL, target='L')

        explain_input = torch.stack((dispL, dispR, dispL, dispL - dispLrec), dim=1) / self.max_disp
        explain_output = self.explain(explain_input)

        return dispL, dispLrec, dispR, explain_output
