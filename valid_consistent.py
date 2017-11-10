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
import data.factory as dfac
from training.util import precision
from data.util import store_disparity
from models.sampler import reconstruct
import models

if __name__ == '__main__':
    directory = sys.argv[1]
    model_dir = os.path.join(directory, 'checkpoints')
    config_file = os.path.join(directory, 'config.json')

    config = conf.Configuration(config_file)

    unary = models.unary.SeLuResnetUnary(**config.config)
    volume = models.volume.CostVolumeDot(**config.config)
    regression = models.regression.SeLuResnetRegression(**config.config)
    classification = models.classification.SoftArgminOclussion(**config.config)

    model = models.stereo.SeLuConsistencyStereoRegression(unary, volume, regression, classification)
    
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(model_dir, config.checkpoint))['state_dict'])

    dataset_splits = dfac.get_dataset_eval(config)

    f = open(os.path.join(directory, 'valid.txt'), 'w')
    sys.stdout = Logger(sys.stdout, f)

    model.cuda()
    model.eval()

    dataloaders = {}
    for split in dataset_splits.keys():
        if len(dataset_splits[split]) > 0:
            dataloaders[split] = DataLoader(dataset_splits[split], batch_size=1, shuffle=False, pin_memory=True)

    saver_pool = Pool(processes=1)

    for split in dataloaders:
        rec_dir = os.path.join(directory, split)
        os.makedirs(rec_dir, exist_ok=True)
        hit_total, total = 0, 0
        dataloader = dataloaders[split]
        for i, example in enumerate(dataloader):
            l, r = Variable(example['left'], volatile=True, requires_grad=False), Variable(example['right'],
                                                                                           volatile=True,
                                                                                           requires_grad=False)
            # start = time()
            lc, rc = l.cuda(async=True), r.cuda(async=True)

            dispL, dispLrec, dispR, explain = model(lc, rc)
            D = dispL.data.cpu().numpy().squeeze()

            if i == 0:
                DL, DLrec, DR, expMask = [x.data.cpu().numpy().squeeze() for x in
                                          (dispL, dispLrec, dispR, F.sigmoid(explain))]
                print(expMask.mean())

                plot_this = [(DL, 'disp left'), (DLrec, 'left from right'), (DR, 'disp right'),
                             (expMask, 'occlusion mask')]
                fig = plt.figure()
                for i, (mat, name) in enumerate(plot_this):
                    ax = fig.add_subplot(len(plot_this), 1, 1 + i)
                    ax.set_title(name)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.imshow(mat, label='')
                plt.show()
            # continue

            # end = time()
            # print("Elapsed time: {}ms".format(1000 * (end - start)), file=sys.stderr)

            store_path = os.path.join(rec_dir, example['name'][0])
            print(store_path, file=sys.stderr)
            saver_pool.apply_async(store_disparity, [D, store_path])

            if 'disparity' in example:
                gt = example['disparity']
                hit, miss = precision(gt.numpy().squeeze(), D, max_disp=config.max_disp)
                hit_total += hit
                total += (hit + miss)
                print('{}: {}/{} {}%'.format(split, hit, (hit + miss), 100 * (hit / (hit + miss))))
        if total > 0:
            print("{}: {}%".format(split, 100 * (hit_total / total)))
