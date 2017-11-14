import torch
import torch.nn as nn


class CostVolumeConcat(nn.Module):
    def __init__(self, **config):
        super(CostVolumeConcat, self).__init__()
        self.stem_stride = config.get('stem_strides', 2)
        self.max_disp = config.get('max_disp', 192)

    def forward(self, un_l, un_r, direction=1):
        return torch.stack(
            [torch.cat([un_l, un_r], dim=1)] + [
                torch.cat((un_l, torch.cat((un_r[:, :, :, -d * direction:], un_r[:, :, :, 0:-d * direction]), dim=3)),
                          dim=1) for d in
                range(1, self.max_disp // self.stem_stride)], dim=2)


class CostVolumeDot(nn.Module):
    def __init__(self, **config):
        super(CostVolumeDot, self).__init__()
        self.stem_stride = config.get('stem_strides', 2)
        self.max_disp = config.get('max_disp', 192)

    def dot(self, L, R, dim):
        return (L * R).sum(dim, keepdim=True)

    def forward(self, un_l, un_r, direction=1):
        return torch.stack(
            [self.dot(un_l, un_r, dim=1)] + [
                self.dot(un_l, torch.cat((un_r[:, :, :, -d * direction:], un_r[:, :, :, 0:-d * direction]), dim=3),
                         dim=1) for d in
                range(1, self.max_disp // self.stem_stride)], dim=2)
