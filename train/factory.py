import torch.optim as optim

from . import loss

losses = {
    'L1sparse': loss.L1LossSparse,
    'L1dense': loss.L1LossDense,
}

def get_loss(config):
    return losses[config.loss]

optimizers = {
    'adam': optim.Adam,
}

def get_optimizer(config):
    return optimizers[config.optimizer]