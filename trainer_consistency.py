from torch.utils.data import DataLoader
import torch.nn as nn

import sys
import os

import experiment.configuration as conf
from experiment.logger import Logger
import models
import data.factory as dfac
import training.factory as tfac
from training.trainer import Trainer
from training.util import num_params
from training.loss import L1SparseConsistencyLoss


if __name__ == '__main__':
    directory = sys.argv[1]
    model_dir = os.path.join(directory, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    config_file = os.path.join(directory, 'config.json')

    config = conf.Configuration(config_file)

    unary = models.unary.SeLuResnetUnary(**config.config)
    volume = models.volume.CostVolumeDot(**config.config)
    regression = models.regression.SeLuResnetRegression(**config.config)
    classification = models.classification.SoftArgminOclussion(**config.config)

    model = models.stereo.SeLuConsistencyStereoRegression(unary, volume, regression, classification)

    print(model)
    print('Number of parameters: {}'.format(num_params(model)))

    criterion = L1SparseConsistencyLoss()
    model.set_criterion(criterion)
    optimizer = tfac.get_optimizer(config)(model.parameters(), lr=config.lr)

    dataset_splits = dfac.get_dataset(config)

    dataloader_train = DataLoader(dataset_splits['train'], batch_size=config.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    dataloaders = {}
    for split in ['train_valid', 'valid', 'test']:
        if len(dataset_splits[split]) > 0:
            dataloaders[split] = DataLoader(dataset_splits[split], batch_size=1, shuffle=False, pin_memory=True)

    f = open(os.path.join(directory, 'train.txt'), 'a')
    sys.stdout = Logger(sys.stdout, f)

    model.cuda()
    model.train()

    with Trainer(model, dataloader_train, dataloaders, optimizer, model_dir, config) as trainer:
        trainer.train()
