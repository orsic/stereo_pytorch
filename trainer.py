import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
import os

import numpy as np

import experiment.configuration as conf
from experiment.logger import Logger
import models.factory as mfac
import data.factory as dfac
import train.factory as tfac


def load(example, train):
    l, r, d = example['left'].cuda(), example['right'].cuda(), example['disparity'].cuda()
    if train:
        return Variable(l, requires_grad=True), Variable(r, requires_grad=True), Variable(d)
    return Variable(l, volatile=True), Variable(r, volatile=True), Variable(d, volatile=True)


if __name__ == '__main__':
    directory = sys.argv[1]
    model_dir = os.path.join(directory, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    config_file = os.path.join(directory, 'config.json')

    config = conf.Configuration(config_file)

    model = mfac.create_model(config)

    model.cuda()

    criterion = nn.L1Loss()
    optimizer = tfac.get_optimizer(config)(model.parameters())

    dataset_splits = dfac.get_dataset(config)

    dataloader_train = DataLoader(dataset_splits['train'], batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    dataloaders = {}

    for split in ['train_valid', 'valid', 'test']:
        if len(dataset_splits[split]) > 0:
            dataloaders[split] = DataLoader(dataset_splits[split], batch_size=1, shuffle=False, num_workers=1,
                                            pin_memory=True)

    if 'train_valid' in dataloaders:
        best = 10000.0

    f = open(os.path.join(directory, 'train.txt'), 'w')
    sys.stdout = Logger(sys.stdout, f)

    model.train()

    for epoch in range(config.epochs):
        try:
            # train steps for epoch
            for example in dataloader_train:
                lc, rc, dc = load(example, True)
                optimizer.zero_grad()
                outputs = model(lc, rc)
                loss = criterion(outputs, dc)
                loss.backward()
                optimizer.step()
                print("train: epoch {} loss {}".format(epoch, loss.data[0]))
            # validation steps for epoch
            # model.eval()
            for split in dataloaders:
                loader = dataloaders[split]
                uses_train_valid = 'train_valid' == split
                if uses_train_valid:
                    losses = []
                for example in loader:
                    lc, rc, dc = load(example, False)
                    optimizer.zero_grad()
                    outputs = model(lc, rc)
                    loss = criterion(outputs, dc)
                    if uses_train_valid:
                        losses.append(loss.data[0])
                    print("{}: epoch {} loss {}".format(split, epoch, loss.data[0]))
                if uses_train_valid:
                    train_valid_epoch_mean = float(np.mean(losses))
                    if train_valid_epoch_mean > best:
                        best = train_valid_epoch_mean
                        torch.save(model, os.path.join(model_dir, '{}.th'.format(epoch)))
        except KeyboardInterrupt:
            torch.save(model, os.path.join(model_dir, 'interrupted_{}.th'.format(epoch)))
            break
    torch.save(model, os.path.join(model_dir, 'final.th'))
    # L = l.cpu().numpy().squeeze().transpose((1, 2, 0))
