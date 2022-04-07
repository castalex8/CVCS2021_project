import os
from typing import List

import torch
import torchvision

from road_signs.German.dataset.GermanTrafficSignDatasetAbs import TRANSFORMS as GERMAN_TRANSFORM, \
    GermanTrafficSignDatasetAbs, CLASSES as GERMAN_CLASSES
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES, MapillaryDatasetAbs
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import TRANSFORMS as MAPILLARY_TRANSFORM
from road_signs.Unknown.dataset.UnknownDatasetAbs import CLASSES as UNKNOWN_CLASSES
from road_signs.Unknown.dataset.UnknownDatasetAbs import TRANSFORMS as UNKNOWN_TRANSFORM, UnknownDatasetAbs

TEST_IMG = 'pedestrian.jpg'
DATASET = os.getenv('DATASET')
RETRIEVAL_IMAGES_DIR = os.path.join(os.getenv('RETRIEVAL_IMAGES_DIR'), DATASET)


datasets = {
    'mapillary': {
        'transform': MAPILLARY_TRANSFORM,
        'dataset': MapillaryDatasetAbs(train=True),
        'class_weights': '0021.pth',
        'retrieval_siamese_weights': '0013.pth',
        'retrieval_triplet_weights': '0017.pth',
        'get_images': lambda x: x.labels,
        'get_image': lambda x: x,
        'get_label': lambda x: x['label'],
        'classes': MAPILLARY_CLASSES,
        'get_image_from_path': lambda retr_img: [
            img for img in datasets[DATASET]['get_images'](datasets[DATASET]['dataset'])
            if retr_img.split('___')[-1] in img['path']
        ][0]
    },
    'german': {
        'transform': GERMAN_TRANSFORM,
        'dataset': GermanTrafficSignDatasetAbs(train=True),
        'class_weights': '0004.pth',
        'retrieval_siamese_weights': '0005.pth',
        'retrieval_triplet_weights': '0018.pth',
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
        'class_weights': '0020.pth',
        'retrieval_siamese_weights': '0016.pth',
        'retrieval_triplet_weights': '0014.pth',
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


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dataset(dataset=DATASET) -> dict:
    return datasets[dataset]


def get_weights(task: str = 'retrieval_siamese', dataset=DATASET) -> str:
    if task == 'retrieval_siamese':
        return os.path.join('road_signs', 'weights', datasets[dataset]['retrieval_siamese_weights'])
    elif task == 'retrieval_triplet':
        return os.path.join('road_signs', 'weights', datasets[dataset]['retrieval_triplet_weights'])
    elif task == 'classification':
        return os.path.join('road_signs', 'weights', datasets[dataset]['class_weights'])


def get_formatted_test_image(dataset=DATASET) -> torchvision.io.image:
    return datasets[dataset]['transform'](
        torchvision.io.read_image(TEST_IMG).float()
    ).reshape([1, 3, 32, 32]).to(get_device())


def get_formatted_image(img: torchvision.io.image, dataset=DATASET) -> torchvision.io.image:
    return datasets[dataset]['transform'](img.float()).reshape([1, 3, 32, 32]).to(get_device())


def get_predicted_class(prediction, dataset=DATASET):
    return datasets[dataset]['classes'][prediction]


def get_retrieval_images() -> List[str]:
    return [os.path.join(RETRIEVAL_IMAGES_DIR, f) for f in os.listdir(RETRIEVAL_IMAGES_DIR)]


def get_image_from_path(ds: dict, img: str) -> torchvision.io.image:
    return ds['transform'](
        ds['dataset'].read_image(ds['get_image'](ds['get_image_from_path'](img))).float()
    ).reshape([1, 3, 32, 32]).to(get_device())


def update_losses(l, losses, max_results, label_retr_img):
    if len(losses) < max_results:
        losses.append((l, label_retr_img))
    else:
        losses = sorted(losses, key=lambda x: x[0])
        if l < losses[-1][0]:
            losses[-1] = (l, label_retr_img)

    return losses


def get_embedding_path(net: str = 'siamese', dataset=DATASET) -> str:
    return os.path.join(os.getenv('RETRIEVAL_EMBEDDING_DIR'), net, dataset)
