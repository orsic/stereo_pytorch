import numpy as np
import torch


def precision(gt, d, max_disp=192, max_error=3):
    sparse_mask = gt > 0
    max_mask = gt < max_disp
    mask = np.logical_and(sparse_mask, max_mask)
    miss = np.sum(np.abs(gt[mask] - d[mask]) > max_error)
    hit = np.sum(np.abs(gt[mask] - d[mask]) <= max_error)
    return hit, miss


def precision_th(gt, d, max_disp=192, max_error=3, return_diff=False):
    sparse_mask = gt > 0
    max_mask = gt < max_disp
    mask = sparse_mask & max_mask
    diff = torch.abs(gt - d)
    miss = diff > max_error
    hit = diff <= max_error
    n_hits, n_misses = torch.sum(hit[mask]), torch.sum(miss[mask])
    if return_diff:
        assert d.size(0) == 1
        R = diff.new(diff.size()[1:3]).int().zero_()
        G = diff.new(diff.size()[1:3]).int().zero_()
        B = diff.new(diff.size()[1:3]).int().zero_()
        R[mask & miss] = 255
        G[mask & hit] = 255
        return n_hits, n_misses, torch.stack((R,G,B), dim=-1)
    return n_hits, n_misses


def num_params(model):
    params = model.parameters()
    total = 0
    for w in params:
        current = 1
        for si in w.size():
            current *= si
        total += current
    return total
