import torch
import numpy as np

from data.util import padding

class Padding(object):

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, sample):
        l, r = sample['left'], sample['right']
        trans = {
            'left': padding(l, self.target_size),
            'right': padding(r, self.target_size),
            'name': sample['name'],
        }
        if 'disparity' in sample:
            trans['disparity'] = padding(sample['disparity'], self.target_size[:2])
        return trans

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, top=None, left=None):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.top = top
        self.left = left

    def __call__(self, sample):
        l, r = sample['left'], sample['right']

        h, w = l.shape[:2]
        new_h, new_w = self.output_size

        if self.top is None and self.left is None:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top = self.top
            left = self.left

        l = l[top: top + new_h, left: left + new_w]
        r = r[top: top + new_h, left: left + new_w]
        trans = {
            'left': l, 'right': r, 'name': sample['name'],
        }
        if 'disparity' in sample:
            d = sample['disparity']
            d = d[top: top + new_h, left: left + new_w]
            trans['disparity'] = d

        return trans


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        l, r = sample['left'], sample['right']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        l = torch.from_numpy(np.ascontiguousarray(l.transpose((2, 0, 1))))
        r = torch.from_numpy(np.ascontiguousarray(r.transpose((2, 0, 1))))
        trans = {
            'left': l, 'right': r, 'name': sample['name'],
        }
        if 'disparity' in sample:
            d = sample['disparity']
            d = torch.from_numpy(np.ascontiguousarray(d))
            trans['disparity'] = d
        return trans
