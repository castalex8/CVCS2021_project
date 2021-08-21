import os
import torch
from torchvision.io import read_image
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES, MapillaryDatasetAbs
from road_signs.Unknown.dataset.UnknownDatasetAbs import CLASSES as UNKNOWN_CLASSES
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import TRANSFORMS as MAPILLARY_TRANSFORM
from road_signs.German.dataset.GermanTrafficSignDatasetAbs import TRANSFORMS as GERMAN_TRANSFORM, GermanTrafficSignDatasetAbs, get_classes
from road_signs.Unknown.dataset.UnknownDatasetAbs import TRANSFORMS as UNKNOWN_TRANSFORM, UnknownDatasetAbs


TEST_IMG = 'pedestrian.png'
DATASET = 'mapillary'


datasets = {
    'mapillary': {
        'transform': MAPILLARY_TRANSFORM,
        'dataset': MapillaryDatasetAbs(train=True),
        'weights': '0012.pth',
        'get_images': lambda x: x.labels,
        'get_image': lambda x: x,
        'get_label': lambda x: x['label'],
        'classes': MAPILLARY_CLASSES
    },
    'german': {
        'transform': GERMAN_TRANSFORM,
        'dataset': GermanTrafficSignDatasetAbs(train=True),
        'weights': '0004.pth',
        'get_images': lambda x: x.img_labels.values,
        'get_image': lambda x: x[-1],
        'get_label': lambda x: get_classes()[x[-2]],
        'classes': get_classes()
    },
    'unknown': {
        'transform': UNKNOWN_TRANSFORM,
        'dataset': UnknownDatasetAbs(train=True),
        'weights': '0011.pth',
        'get_images': lambda x: x.labels,
        'get_image': lambda x: x,
        'get_label': lambda x: x['label'],
        'classes': UNKNOWN_CLASSES
    }
}


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dataset():
    return datasets[DATASET]


def get_weights():
    return os.path.join('weights', datasets[DATASET]['weights'])


def get_formatted_test_image():
    return datasets[DATASET]['transform'](read_image(TEST_IMG).float()).reshape([1, 3, 32, 32])


def get_formatted_image(img):
    return datasets[DATASET]['transform'](img.float()).reshape([1, 3, 32, 32])


def get_predicted_class(prediction):
    return datasets[DATASET][prediction]
