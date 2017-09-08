import torch
import torch.nn as nn


class CostVolumeConcat(nn.Module):
    def __init__(self, **config):
        super(CostVolumeConcat, self).__init__()
        self.stem_stride = config.get('stem_stride', 2)
        self.max_disp = config.get('max_disp', 192)

    def forward(self, un_l, un_r):
        return torch.stack(
            [torch.cat([un_l, un_r], dim=1)] + [
                torch.cat((un_l, torch.cat((un_r[:, :, :, -d:], un_r[:, :, :, 0:-d]), dim=3)), dim=1) for d in
                range(1, self.max_disp // self.stem_stride)], dim=2)
