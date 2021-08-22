import os
import torch
from torchvision.io import read_image
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES, MapillaryDatasetAbs
from road_signs.Unknown.dataset.UnknownDatasetAbs import CLASSES as UNKNOWN_CLASSES
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import TRANSFORMS as MAPILLARY_TRANSFORM
from road_signs.German.dataset.GermanTrafficSignDatasetAbs import TRANSFORMS as GERMAN_TRANSFORM, \
    GermanTrafficSignDatasetAbs, CLASSES as GERMAN_CLASSES
from road_signs.Unknown.dataset.UnknownDatasetAbs import TRANSFORMS as UNKNOWN_TRANSFORM, UnknownDatasetAbs


TEST_IMG = 'pedestrian.jpg'
DATASET = 'mapillary'
RETRIEVAL_IMAGES_DIR = os.path.join('road_signs', 'retrieval_images', DATASET)


datasets = {
    'mapillary': {
        'transform': MAPILLARY_TRANSFORM,
        'dataset': MapillaryDatasetAbs(train=True),
        'class_weights': '0012.pth',
        'retrieval_siamese_weights': '0017.pth',
        'get_images': lambda x: x.labels,
        'get_image': lambda x: x,
        'get_label': lambda x: x['label'],
        'classes': MAPILLARY_CLASSES,
        'get_image_from_path': lambda retr_img: [
            img for img in datasets[DATASET]['get_images'](datasets[DATASET]['dataset'])
            if retr_img.split('__')[-1] in img['path']
         ][0]
    },
    'german': {
        'transform': GERMAN_TRANSFORM,
        'dataset': GermanTrafficSignDatasetAbs(train=True),
        'class_weights': '0004.pth',
        'retrieval_siamese_weights': '0005.pth',
        'get_images': lambda x: x.img_labels.values,
        'get_image': lambda x: x[-1],
        'get_label': lambda x: GERMAN_CLASSES[x[-2]],
        'classes': GERMAN_CLASSES,
        'get_image_from_path': lambda retr_img: [
            img for img in datasets[DATASET]['get_images'](datasets[DATASET]['dataset'])
            if retr_img.split('__')[-1] in img[-1]
         ][0]
    },
    'unknown': {
        'transform': UNKNOWN_TRANSFORM,
        'dataset': UnknownDatasetAbs(train=True),
        'class_weights': '0011.pth',
        'retrieval_siamese_weights': '0016.pth',
        'get_images': lambda x: x.labels,
        'get_image': lambda x: x,
        'get_label': lambda x: x['label'],
        'classes': UNKNOWN_CLASSES,
        'get_image_from_path': lambda retr_img: [
            img for img in datasets[DATASET]['get_images'](datasets[DATASET]['dataset'])
            if retr_img.split('__')[-1] in img['path']
         ][0]
    }
}


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dataset():
    return datasets[DATASET]


def get_weights(task='retrieval_siamese'):
    return os.path.join('road_signs', 'weights', datasets[DATASET]['class_weights']) if task == 'classification' else \
        os.path.join('road_signs', 'weights', datasets[DATASET]['retrieval_siamese_weights'])


def get_formatted_test_image():
    return datasets[DATASET]['transform'](read_image(TEST_IMG).float()).reshape([1, 3, 32, 32])


def get_formatted_image(img):
    return datasets[DATASET]['transform'](img.float()).reshape([1, 3, 32, 32])


def get_predicted_class(prediction):
    return datasets[DATASET]['classes'][prediction]


def get_retrieval_images():
    return [os.path.join(RETRIEVAL_IMAGES_DIR, f) for f in os.listdir(RETRIEVAL_IMAGES_DIR)]
