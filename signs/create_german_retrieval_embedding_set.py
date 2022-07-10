import os
import pathlib

import torch
from torchvision.io import read_image

from signs.road_signs.datasets_utils import get_formatted_image
from signs.road_signs.siamese_retrieval import get_embedding_from_img as get_siamese_embedding
from signs.road_signs.triplet_retrieval import get_embedding_from_img as get_triplet_embedding

NUMBER_PICTURES = 20

RETRIEVAL_IMAGES_DIR = os.getenv('RETRIEVAL_IMAGES_DIR')
RETRIEVAL_EMBEDDING_DIR = os.getenv('RETRIEVAL_EMBEDDING_DIR')


def exists_or_create():
    for p_folder in ['siamese', 'triplet']:
        folder = os.path.join(RETRIEVAL_EMBEDDING_DIR, p_folder, 'german')
        if not pathlib.Path(folder).exists():
            os.mkdir(folder)


def create_german_ds():
    # Create German retrieval set
    exists_or_create()
    print('Creating german dataset... ', end='')
    base_image_path = os.path.join(RETRIEVAL_IMAGES_DIR, 'german.eval')

    for img_path in os.listdir(base_image_path):
        base_path = os.path.join(base_image_path, img_path)
        img = get_formatted_image(read_image(base_path), dataset='german')
        embedding, _ = get_siamese_embedding(img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'siamese', 'german', os.path.basename(base_path)[:-3] + 'pt'))
        embedding, _, _ = get_triplet_embedding(img, img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'triplet', 'german', os.path.basename(base_path)[:-3] + 'pt'))

    print('done!')


if __name__ == '__main__':
    create_german_ds()
