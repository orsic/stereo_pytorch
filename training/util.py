import numpy as np
import torch


def precision(gt, d, max_disp=192, max_error=3):
    sparse_mask = gt > 0
    max_mask = gt < max_disp
    mask = np.logical_and(sparse_mask, max_mask)
    miss = np.sum(np.abs(gt[mask] - d[mask]) > max_error)
    hit = np.sum(np.abs(gt[mask] - d[mask]) <= max_error)
    return hit, miss


def precision_th(gt, d, max_disp=192, max_error=3):
    sparse_mask = gt > 0
    max_mask = gt < max_disp
    mask = sparse_mask & max_mask
    miss = torch.sum(torch.abs(gt[mask] - d[mask]) > max_error)
    hit = torch.sum(torch.abs(gt[mask] - d[mask]) <= max_error)
    return hit, miss


def num_params(model):
    params = model.parameters()
    total = 0
    for w in params:
        current = 1
        for si in w.size():
            current *= si
        total += current
    return total

