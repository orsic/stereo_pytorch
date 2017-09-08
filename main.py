import torch
from torch.autograd import Variable
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from models.stereo import StereoRegression
from models.unary import ResnetUnary
from models.volume import CostVolumeConcat
from models.regression import ResnetRegression
from models.classification import SoftArgmin

from data.dataset_paths import get_all_kitti_paths
from data.dataset import KittiDataset
from data.transform import Padding, RandomCrop, ToTensor

model = StereoRegression(
    ResnetUnary(),
    CostVolumeConcat(),
    ResnetRegression(),
    SoftArgmin(),
)

size = (1, 3, 512, 256)
model.cuda()

# disparity = out.data.cpu().numpy().squeeze()

transform = Compose((RandomCrop((256, 512)), ToTensor()))

dataset = KittiDataset(get_all_kitti_paths(), transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=4)

example = dataset[0]
l, r, d = example['left'], example['right'], example['disparity']

disp = model(Variable(l.unsqueeze_(0).cuda()), Variable(r.unsqueeze_(0).cuda()))
