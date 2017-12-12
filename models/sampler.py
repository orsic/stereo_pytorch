import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def create_meshgrid(N, H, W):
    return Variable(torch.arange(0, H * W).view(1, H, W).repeat(N, 1, 1).long(), requires_grad=False).cuda(async=True)


def shift_disparity(grid, disparity, direction):
    shift = direction * disparity
    return torch.clamp(grid + shift, 0, torch.numel(disparity.data) - 1)


def reconstruct(image, disparity, target='L'):
    assert target in ['L', 'R']

    shift = -1 if target == 'L' else 1

    if len(image.size()) == 4:
        N, C, H, W = image.size()
        channels = [image[:, c, :, :] for c in range(C)]
    else:
        N, H, W = image.size()
        channels = [image]

    channel_transforms = []
    index_grid = create_meshgrid(N, H, W)
    disp_flat = disparity.view(-1)
    dint_floor = torch.floor(disparity).long()
    dint_ceil = torch.ceil(disparity).long()

    index_shift_floor = shift_disparity(index_grid, dint_floor, shift).view(-1).detach()
    index_shift_ceil = shift_disparity(index_grid, dint_ceil, shift).view(-1).detach()

    for i, channel in enumerate(channels):
        flat = channel.view(-1)
        alpha = 1.0 - (disp_flat - dint_floor.float())

        rec_flat_floor = flat[index_shift_floor]
        rec_flat_ceil = flat[index_shift_ceil]

        rec_flat = rec_flat_floor * alpha + rec_flat_ceil * (1.0 - alpha)
        channel_transforms.append(rec_flat.view(N, H, W))
    if len(image.size()) == 4:
        return torch.stack(channel_transforms, dim=1)
    else:
        return channel_transforms[0]


def sampler_reconstruct(image, disparity, target='L'):
    assert len(disparity.size()) == 3
    assert target in ['L', 'R']

    if len(image.size()) == 3:
        image = image.unsqueeze(1)

    disparity = torch.unsqueeze(disparity, dim=-1).data

    zeros = torch.zeros(disparity.size()).float().cuda(async=True)

    grid = torch.cat((disparity / disparity.max(), zeros), dim=-1)

    sampled = F.grid_sample(image, grid)

    return sampled


def sampler_flow(image, flow):
    N, C, H, W = image.size()
    grid_dims = [torch.from_numpy(m) for m in np.meshgrid(np.linspace(-1, 1, W, dtype=np.float32), np.linspace(-1, 1, H, dtype=np.float32))]
    grid = torch.stack(grid_dims, dim=-1).unsqueeze(0).repeat(N,1,1,1)
    grid[...,0] += flow[...,0] / H
    grid[...,1] += flow[...,1] / W
    return F.grid_sample(image, grid)
