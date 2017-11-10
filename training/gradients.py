import torch

import os
import pickle


def parameter_norms(optimizer):
    '''
    :param optimizer: torch.optim.Optimizer
    :return: squared L2 norm of all model gradients: float
    '''
    grad_norms = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data.view(-1)
            d_norm = torch.sum(d_p * d_p)
            grad_norms.append(d_norm)
    return sum(grad_norms)


def store_parameter_norms(norms, directory, name='param_norms.pkl'):
    path = os.path.join(directory, name)
    with open(path, 'wb') as file:
        pickle.dump(norms, file)
