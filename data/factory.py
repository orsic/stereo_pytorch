from torchvision.transforms import Compose

from . import dataset, dataset_paths, transform, util

dataset_path_creators = {
    'kitti': dataset_paths.get_all_kitti_paths,
    'cityscapes': dataset_paths.get_cityscapes_paths,
    'kitti_submission': dataset_paths.get_all_kitti_test_paths,
    'sceneflow': dataset_paths.get_all_sceneflow_paths,
    'odometry': dataset_paths.get_odometry_paths,
}


def get_paths_for_dataset(config):
    return dataset_path_creators[config.dataset]()


transform_classes = {
    'random_crop': transform.RandomCrop,
    'tensor': transform.ToTensor,
    'padding': transform.Padding,
}


def get_transforms(transforms):
    return Compose([transform_classes[trans[0]](*trans[1:]) for trans in transforms])


def get_dataset_eval(config):
    dataset_classes = {
        'kitti': dataset.KittiDataset,
        'sceneflow': dataset.SceneFlowDataset,
        'cityscapes': dataset.CityScapes,
    }
    paths = get_paths_for_dataset(config)
    if config.dataset == 'cityscapes':
        split = paths
    else:
        split = util.split_dataset_paths(paths, config.train_ratio, config.train_valid_ratio, config.valid_ratio,
                                     config.test_ratio)
    transformations = [transform.ToTensor()]
    if config.dataset == 'kitti':
        transformations.insert(0, transform.Padding((384, 1248, 3)))

    transforms = {
        key: Compose(transformations) for key in split
    }
    return {phase: dataset_classes[config.dataset](split[phase], transforms[phase]) for phase in split}


def get_dataset(config):
    dataset_classes = {
        'kitti': dataset.KittiDataset,
        'sceneflow': dataset.SceneFlowDataset,
    }
    paths = get_paths_for_dataset(config)
    split = util.split_dataset_paths(paths, config.train_ratio, config.train_valid_ratio, config.valid_ratio,
                                     config.test_ratio)
    transforms = {
        key: get_transforms(config.dataset_transform[key]) for key in split
    }
    return {phase: dataset_classes[config.dataset](split[phase], transforms[phase]) for phase in split}
