import json
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import transforms

from road_signs.datasets_utils import get_formatted_image
from road_signs.siamese_retrieval import get_embedding_from_img as get_siamese_embedding
from road_signs.triplet_retrieval import get_embedding_from_img as get_triplet_embedding

NUMBER_PICTURES = 20

RETRIEVAL_IMAGES_DIR = os.getenv('RETRIEVAL_IMAGES_DIR')
RETRIEVAL_EMBEDDING_DIR = os.getenv('RETRIEVAL_EMBEDDING_DIR')
UNKNOWN_PATH = os.getenv('UNKNOWN_BASE_DIR')
MAPILLARY_PATH = os.getenv('MAPILLARY_BASE_DIR')


def clean_ds():
    for p_folder in ['siamese', 'triplet']:
        for folder in ['german', 'unknown', 'mapillary']:
            if pathlib.Path(folder).exists():
                shutil.rmtree(os.path.join(RETRIEVAL_EMBEDDING_DIR, p_folder, folder))


def exists_or_create(ds: str):
    for p_folder in ['siamese', 'triplet']:
        folder = os.path.join(RETRIEVAL_EMBEDDING_DIR, p_folder, ds)
        if not pathlib.Path(folder).exists():
            os.mkdir(folder)


def create_german_ds():
    # Create German retrieval set
    exists_or_create('german')
    print('Creating german dataset... ', end='')
    base_image_path = os.path.join(RETRIEVAL_IMAGES_DIR, 'german')

    for img_path in os.listdir(base_image_path):
        base_path = os.path.join(base_image_path, img_path)
        img = get_formatted_image(read_image(base_path), dataset='german')
        embedding, _ = get_siamese_embedding(img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'siamese', 'german', os.path.basename(base_path)[:-3] + 'pt'))
        embedding, _, _ = get_triplet_embedding(img, img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'triplet', 'german', os.path.basename(base_path)[:-3] + 'pt'))

    print('done!')


def create_unknown_ds():
    # Create Unknown retrieval set
    exists_or_create('unknown')
    print('Creating unknown dataset... ', end='')
    base_image_path = os.path.join(RETRIEVAL_IMAGES_DIR, 'unknown')
    for picture in os.listdir(base_image_path):
        base_name = picture.split('__')[-1][:-4]
        base_path = os.path.join(base_image_path, picture)
        ann_doc = ET.parse(os.path.join(UNKNOWN_PATH, 'annotations', f"{base_name}.xml")).getroot()
        image = transforms.ToTensor()(cv2.imread(base_path)[:, :, :3])
        img = image[
            :,
            int(ann_doc.find("./object/bndbox/ymin").text):int(ann_doc.find("./object/bndbox/ymax").text),
            int(ann_doc.find("./object/bndbox/xmin").text):int(ann_doc.find("./object/bndbox/xmax").text)
        ].float()

        img = get_formatted_image(img)
        embedding, _ = get_siamese_embedding(img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'siamese', 'unknown', os.path.basename(base_path)[:-3] + 'pt'))
        embedding, _, _ = get_triplet_embedding(img, img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'triplet', 'unknown', os.path.basename(base_path)[:-3] + 'pt'))

    print('done!')


def create_mapillary_ds():
    # Create Mapillary retrieval set
    exists_or_create('mapillary')
    print('Creating unknown mapillary... ', end='')
    base_image_path = os.path.join(RETRIEVAL_IMAGES_DIR, 'mapillary')
    for picture in os.listdir(base_image_path):
        base_name = picture.split('___')[-1][:-4]
        base_path = os.path.join(base_image_path, picture)
        bb = [
            obj['bbox'] for obj in
            json.load(open(os.path.join(MAPILLARY_PATH, 'annotations', 'annotations', f"{base_name}.json")))['objects']
            if obj['label'] == picture.split('---')[0]
        ][0]

        image = torchvision.io.read_image(base_path)
        img = image[:, int(bb['ymin']):int(bb['ymax']), int(bb['xmin']):int(bb['xmax'])].float()

        img = get_formatted_image(img)
        embedding, _ = get_siamese_embedding(img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'siamese', 'mapillary', os.path.basename(base_path)[:-3] + 'pt'))
        embedding, _, _ = get_triplet_embedding(img, img, img)
        torch.save(embedding, os.path.join(RETRIEVAL_EMBEDDING_DIR, 'triplet', 'mapillary', os.path.basename(base_path)[:-3] + 'pt'))

    print('done!')


def main():
    clean_ds()
    create_german_ds()
    create_unknown_ds()
    create_mapillary_ds()


if __name__ == '__main__':
    main()
