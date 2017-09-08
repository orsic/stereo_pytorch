from torch.utils.data import Dataset

from data.util import open_image, load_disparity


class KittiDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        example = {
            'left': open_image(path['left']),
            'right': open_image(path['right']),
            'disparity': load_disparity(path['disparity'])
        }
        return self.transform(example)


class SceneFlowDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        example = {
            'left': open_image(path['left']),
            'right': open_image(path['right']),
            'disparity': load_disparity(path['disparity'])
        }
        return self.transform(example)
