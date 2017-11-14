import sys
import os
import torch
from time import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from multiprocessing import Pool
import matplotlib.pyplot as plt
import torch.nn.functional as F

import experiment.configuration as conf
from experiment.logger import Logger
import models.factory as mfac
import data.factory as dfac
from training.util import precision_th, precision
from data.util import store_disparity
from models.sampler import reconstruct, sampler_reconstruct

if __name__ == '__main__':
    directory = sys.argv[1]
    model_dir = os.path.join(directory, 'checkpoints')
    config_file = os.path.join(directory, 'config.json')

    config = conf.Configuration(config_file)

    model = mfac.create_model(config)
    model.load_state_dict(torch.load(os.path.join(model_dir, config.get('checkpoint', 'model_best.pth.tar')))['state_dict'])

    dataset_splits = dfac.get_dataset_eval(config)

    f = open(os.path.join(directory, 'valid.txt'), 'w')
    sys.stdout = Logger(sys.stdout, f)

    model.cuda()
    model.eval()

    dataloaders = {}
    for split in dataset_splits.keys():
        if len(dataset_splits[split]) > 0:
            dataloaders[split] = DataLoader(dataset_splits[split], batch_size=1, shuffle=False,
                                            pin_memory=True)

    saver_pool = Pool(processes=4)

    for split in dataloaders:
        rec_dir = os.path.join(directory, split)
        os.makedirs(rec_dir, exist_ok=True)
        hit_total, total = 0, 0
        hit_total_trash, total_trash = 0, 0
        dataloader = dataloaders[split]
        for i, example in enumerate(dataloader):
            l, r = Variable(example['left'], volatile=True, requires_grad=False), Variable(example['right'],
                                                                                           volatile=True,
                                                                                           requires_grad=False)
            lc, rc = l.cuda(async=True), r.cuda(async=True)

            unary, volume, regression, classification = model.forward_all(lc, rc)
            classification_aux = model.classify(volume, regression.size()[2:])
            D = classification.data.cpu().numpy().squeeze()
            D_aux = classification_aux.data.cpu().numpy().squeeze()

            store_path = os.path.join(rec_dir, example['name'][0])
            store_path_aux = os.path.join(rec_dir, 'aux_' + example['name'][0])

            saver_pool.apply_async(store_disparity, [D, store_path])
            saver_pool.apply_async(store_disparity, [D_aux, store_path_aux])

            if 'disparity' in example:
                gt = example['disparity']
                hit, miss = precision_th(gt.cuda(), classification.data, max_disp=config.max_disp)
                hit_trash, miss_trash = precision_th(gt.cuda(), classification_aux.data, max_disp=config.max_disp)
                hit_total += hit
                hit_total_trash += hit_trash
                total += (hit + miss)
                total_trash += (hit_trash + miss_trash)
                print('{}: {}/{} {}%'.format(split, hit, (hit + miss), 100 * (hit / (hit + miss))))
                print('{}: {}/{} {}%'.format(split, hit_trash, (hit_trash + miss_trash), 100 * (hit_trash / (hit_trash + miss_trash))), file=sys.stderr)
            else:
                print(store_path, file=sys.stderr)
        if total > 0:
            print("{}: {}%".format(split, 100 * (hit_total / total)))
            print("{}: {}%".format(split, 100 * (hit_total_trash / total_trash)), file=sys.stderr)
