import os
import pathlib
import xml.etree.ElementTree as ET

import cv2
import torch
from torchvision.transforms import transforms

from signs.road_signs.datasets_utils import get_formatted_image
from signs.road_signs.siamese_retrieval import get_embedding_from_img as get_siamese_embedding
from signs.road_signs.triplet_retrieval import get_embedding_from_img as get_triplet_embedding

NUMBER_PICTURES = 20

RETRIEVAL_IMAGES_DIR = os.getenv('RETRIEVAL_IMAGES_DIR')
RETRIEVAL_EMBEDDING_DIR = os.getenv('RETRIEVAL_EMBEDDING_DIR')
UNKNOWN_PATH = os.getenv('UNKNOWN_BASE_DIR')


def exists_or_create():
    for p_folder in ['siamese', 'triplet']:
        folder = os.path.join(RETRIEVAL_EMBEDDING_DIR, p_folder, 'unknown')
        if not pathlib.Path(folder).exists():
            os.mkdir(folder)


def create_unknown_ds():
    # Create Unknown retrieval set
    exists_or_create()
    print('Creating unknown dataset... ', end='')
    base_image_path = os.path.join(RETRIEVAL_IMAGES_DIR, 'unknown.eval')

    for img_path in os.listdir(base_image_path):
        base_path = os.path.join(base_image_path, img_path)
        img = get_formatted_image(transforms.ToTensor()(cv2.imread(base_path)), dataset='unknown')
        embedding, _ = get_siamese_embedding(img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'siamese', 'german', os.path.basename(base_path)[:-3] + 'pt'))
        embedding, _, _ = get_triplet_embedding(img, img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'triplet', 'german', os.path.basename(base_path)[:-3] + 'pt'))

    print('done!')


if __name__ == '__main__':
    create_unknown_ds()
