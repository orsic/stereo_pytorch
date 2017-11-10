import torch.nn as nn
import torch.optim as optim

from . import loss

losses = {
    'L1sparse': loss.L1LossSparse,
    'L1dense': loss.L1LossDense,
    'L1': loss.L1Loss,
}

def get_loss(config):
    return losses[config.loss]

optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
}

def get_optimizer(config):
    return optimizers[config.optimizer]