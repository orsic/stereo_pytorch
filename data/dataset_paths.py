import os
import itertools
import random
import config
from collections import namedtuple

SCENE_FLOW_HOME = config.get('SCENE_FLOW_HOME')
SCENE_FLOW_DIR_LEFT = 'left'
SCENE_FLOW_DIR_RIGHT = 'right'
SCENE_FLOW_DIR_DISPARITY = 'disparity'
SCENE_FLOW_DATASETS = [os.path.join(focal, side, speed) for focal, side, speed in
                       itertools.product(
                           ['15mm_focallength', '35mm_focallength'],
                           ['scene_forwards', 'scene_backwards'],
                           ['slow', 'fast'])]


def get_all_sceneflow_paths():
    random.seed(5)
    paths = []
    for dataset in SCENE_FLOW_DATASETS:
        dataset_path = os.path.join(SCENE_FLOW_HOME, dataset)
        dataset_examples = [example[:-4] for example in
                            filter(lambda x: x.endswith('png'),
                                   os.listdir(os.path.join(dataset_path, SCENE_FLOW_DIR_LEFT)))]
        paths.extend([{'left': os.path.join(dataset_path, SCENE_FLOW_DIR_LEFT, example + '.png'),
                       'right': os.path.join(dataset_path, SCENE_FLOW_DIR_RIGHT, example + '.png'),
                       'disparity': os.path.join(dataset_path, SCENE_FLOW_DIR_DISPARITY, example + '.pfm')}
                      for example in dataset_examples])
    random.shuffle(paths)
    return paths


KITTI_HOME = config.get('KITTI_HOME')
KITTI_TRAINING_HOME = os.path.join(KITTI_HOME, 'training')
KITTI_TRAINING_LEFT = os.path.join(KITTI_TRAINING_HOME, 'image_2')
KITTI_TRAINING_RIGHT = os.path.join(KITTI_TRAINING_HOME, 'image_3')
KITTI_TRAINING_DISPARITIES = os.path.join(KITTI_TRAINING_HOME, 'disp_occ_0')


def get_all_kitti_paths():
    random.seed(5)
    paths = [{
        'left': os.path.join(KITTI_TRAINING_LEFT, image_name),
        'right': os.path.join(KITTI_TRAINING_RIGHT, image_name),
        'disparity': os.path.join(KITTI_TRAINING_DISPARITIES, image_name),
    } for image_name in sorted(os.listdir(KITTI_TRAINING_DISPARITIES))]
    random.shuffle(paths)
    return paths


KITTI_ODOMETRY_HOME = config.get('KITTI_ODOMETRY_HOME')


def get_odometry_paths():
    random.seed(5)
    paths = []
    for sequence in sorted(os.listdir(KITTI_ODOMETRY_HOME)):
        left_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_2')
        right_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_3')
        for name in sorted(filter(lambda x: x.endswith('png'), os.listdir(left_folder))):
            paths.append({
                'disparity': None,
                'left': os.path.join(left_folder, name),
                'right': os.path.join(right_folder, name),
            })
    random.shuffle(paths)
    return paths

def get_odometry_paths_demo():
    paths = {}
    for sequence in sorted(os.listdir(KITTI_ODOMETRY_HOME)):
        paths[sequence] = []
        left_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_2')
        right_folder = os.path.join(KITTI_ODOMETRY_HOME, sequence, 'image_3')
        for name in sorted(filter(lambda x: x.endswith('png'), os.listdir(left_folder))):
            paths[sequence].append({
                'disparity': None,
                'left': os.path.join(left_folder, name),
                'right': os.path.join(right_folder, name),
            })
    return paths


KITTI_TESTING_HOME = os.path.join(KITTI_HOME, 'testing')
KITTI_TESTING_LEFT = os.path.join(KITTI_TESTING_HOME, 'image_2')
KITTI_TESTING_RIGHT = os.path.join(KITTI_TESTING_HOME, 'image_3')


def get_all_kitti_test_paths():
    paths = [{
        'left': os.path.join(KITTI_TESTING_LEFT, image_name),
        'right': os.path.join(KITTI_TESTING_RIGHT, image_name),
        'disparity': None,
    } for image_name in sorted(filter(lambda x: x.endswith('png'), os.listdir(KITTI_TESTING_LEFT)))]
    random.shuffle(paths)
    return paths

CITYSCAPES_HOME = config.get('CITYSCAPES_HOME')
CITYSCAPES_LEFT = os.path.join(CITYSCAPES_HOME, 'rgb')
CITYSCAPES_RIGHT = os.path.join(CITYSCAPES_HOME, 'right/rightImg8bit')

CityscapesPath = namedtuple('CityscapesPath', 'split city name left right')


def get_cityscapes_paths():
    def getp(split, city, file):
        name = file.split('.')[0]
        file_r = file[:-4] + '_rightImg8bit.png'
        L, R = os.path.join(CITYSCAPES_LEFT, split, city, file), os.path.join(CITYSCAPES_RIGHT, split, city, file_r)
        return CityscapesPath(split, city, name, L, R)

    splits = os.listdir(CITYSCAPES_LEFT)
    paths = {}
    for split in splits:
        paths[split] = []
        cities = os.listdir(os.path.join(CITYSCAPES_LEFT, split))
        for city in cities:
            filenames = os.listdir(os.path.join(CITYSCAPES_LEFT, split, city))
            for filename in filenames:
                paths[split].append(getp(split, city, filename))
    return paths