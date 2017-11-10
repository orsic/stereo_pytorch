import torch

import os
import shutil
import sys

from data.util import load_variable
from .util import precision_th


def save_checkpoint(state, model_dir, is_best, filename='checkpoint.pth.tar'):
    filepath = os.path.join(model_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(model_dir, 'model_best.pth.tar'))


class Trainer(object):
    def __init__(self, model, dataloader_train, dataloaders, optimizer, model_dir, config):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.epochs = config.get('epochs')
        self.batch_size = config.get('batch_size')
        self.num_steps = len(self.dataloader_train) * self.epochs // self.batch_size
        self.epoch_start = 0
        self.global_step = 0
        self.best_validation = 0.0
        self.lr = config.get('lr')
        self.lr_min = config.get('lr_min')
        self.checkpoint = config.get('checkpoint')
        self.unary_scale_loss = config.get('unary_scale_loss', 0.0)
        self.max_disp = config.get('max_disp', 192)
        self.max_error = config.get('max_error', 3)

    def __enter__(self):
        if self.checkpoint is not None:
            path = os.path.join(self.model_dir, self.checkpoint)
            checkpoint = torch.load(path)
            print("Loading state from {}".format(path), file=sys.stderr)
            self.epoch_start = checkpoint['epoch']
            self.global_step = len(self.dataloader_train) * self.epoch_start // self.batch_size
            self.best_validation = checkpoint['best_validation']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.save(self.model, os.path.join(self.model_dir, 'final.th'))

    def set_learning_rate(self, optimizer, step, lr_decay_power=1.0):
        lr = (self.lr - self.lr_min) * (1 - step / self.num_steps) ** lr_decay_power + self.lr_min
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        for epoch in range(self.epoch_start, self.epochs):
            try:
                # train steps for epoch
                self.model.train()
                for param_group in self.optimizer.param_groups:
                    print('LR: ', param_group['lr'], file=sys.stderr)
                for i, example in enumerate(self.dataloader_train):
                    self.optimizer.zero_grad()
                    self.set_learning_rate(self.optimizer, self.global_step)
                    self.global_step += 1
                    lc, rc, dc = load_variable(example, True)
                    loss = self.model.loss(dc, lc, rc, unary_scale=self.unary_scale_loss)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print("train: epoch {} loss {}".format(epoch, loss.data[0]))
                # validation
                self.model.eval()
                is_best = False
                hits, misses = self.evaluate(self.dataloaders['train_valid'], epoch)
                precision_val = 100 * hits / (hits + misses)
                print("{}: epoch {} total precision {}".format('train_valid', epoch, precision_val))
                if precision_val > self.best_validation:
                    self.best_validation = precision_val
                    is_best = True
                self.save(epoch, is_best)
            except KeyboardInterrupt:
                self.save(epoch, False)
                break

    def save(self, epoch, is_best):
        state = {
            'epoch': epoch + 1,
            'best_validation': self.best_validation,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        save_checkpoint(state, self.model_dir, is_best)

    def evaluate(self, loader_val, epoch):
        hits, misses = 0, 0
        for i, example in enumerate(loader_val):
            lc, rc, gt = load_variable(example, False)
            out = self.model.forward(lc, rc)
            hit, miss = precision_th(gt.data, out.data, self.max_disp, max_error=self.max_error)
            print('{}: epoch {} {}/{} {}%'.format('train_valid', epoch, hit, (hit + miss), 100 * (hit / (hit + miss))))
            hits, misses = hits + hit, misses + miss
        return hits, misses
