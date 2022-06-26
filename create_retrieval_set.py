import json
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
import torchvision
from cv2 import cv2

from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES, NUM_SAMPLES
from road_signs.German.dataset.GermanTrafficSignDatasetAbs import CLASSES as GERMAN_CLASSES

NUMBER_PICTURES = 20
GERMAN_PATH = os.getenv('GERMAN_BASE_DIR')
UNKNOWN_PATH = os.getenv('UNKNOWN_BASE_DIR')
MAPILLARY_PATH = os.getenv('MAPILLARY_BASE_DIR')

RETRIEVAL_IMAGES_DIR = os.getenv('RETRIEVAL_IMAGES_DIR')


def clean_ds():
    for folder in ['german', 'unknown', 'mapillary']:
        if pathlib.Path(folder).exists():
            shutil.rmtree(os.path.join(RETRIEVAL_IMAGES_DIR, folder))


def exists_or_create(ds: str):
    folder = os.path.join(RETRIEVAL_IMAGES_DIR, ds)
    if not pathlib.Path(folder).exists():
        os.mkdir(folder)


def create_german_ds():
    # Create German retrieval set
    exists_or_create('german.eval')
    print('Creating german dataset... ', end='')
    pd_csv = pd.read_csv(os.path.join(GERMAN_PATH, 'Test.csv'))
    minimum_image_per_class = 20
    img_per_class = defaultdict(int)

    for i in range(len(pd_csv)):
        elem = pd_csv.iloc[i]
        image = torchvision.io.read_image(os.path.join(GERMAN_PATH, elem.Path))
        x1 = elem.values[2]
        x2 = elem.values[4]
        y1 = elem.values[3]
        y2 = elem.values[5]
        img = image[:, y1:y2, x1:x2]
        if img_per_class[elem.ClassId] >= minimum_image_per_class:
            continue

        img_per_class[elem.ClassId] += 1
        torchvision.io.write_png(
            img,
            os.path.join(RETRIEVAL_IMAGES_DIR, 'german.eval', f'{elem.ClassId}_{i}__{os.path.basename(elem.Path)}')
        )

    print('done!')


def create_unknown_ds():
    # Create Unknown retrieval set
    exists_or_create('unknown.eval')
    print('Creating unknown dataset... ', end='')
    d = defaultdict(int)
    base_path = os.path.join(UNKNOWN_PATH, 'images')
    for i, picture in enumerate(os.listdir(base_path)):
        if i % 5 == 0:
            base_name = picture[:-4]
            ann_doc = ET.parse(os.path.join(UNKNOWN_PATH, 'annotations', f"{base_name}.xml")).getroot()
            label = ann_doc.find("./object/name").text
            image = cv2.imread(os.path.join(base_path, picture))
            img = image[
                int(ann_doc.find("./object/bndbox/ymin").text):int(ann_doc.find("./object/bndbox/ymax").text),
                int(ann_doc.find("./object/bndbox/xmin").text):int(ann_doc.find("./object/bndbox/xmax").text),
                :
            ]

            d[label] += 1
            cv2.imwrite(os.path.join(RETRIEVAL_IMAGES_DIR, 'unknown.eval', f"{label}-{d[label]}__{picture}"), img)
    print('done!')


def create_mapillary_ds():
    # Create Mapillary retrieval set
    exists_or_create('mapillary.eval')
    print('Creating mapillary... ', end='')
    d = defaultdict(int)
    base_path = os.path.join(MAPILLARY_PATH, 'images.eval')

    minimum_image_per_class = 20
    img_per_class = defaultdict(int)

    for picture in os.listdir(base_path):
        base_name = picture[:-4]
        annotations = [obj for obj in json.load(
            open(os.path.join(MAPILLARY_PATH, 'annotations', 'annotations', f"{base_name}.json"))
        )['objects'] if obj['label'] in MAPILLARY_CLASSES]

        for ann in annotations:
            cls = ann['label']
            bb = ann['bbox']
            image = torchvision.io.read_image(os.path.join(base_path, base_name + '.jpg'))
            image = image[:, int(bb['ymin']):int(bb['ymax']), int(bb['xmin']):int(bb['xmax'])]

            if img_per_class[cls] >= minimum_image_per_class:
                continue

            img_per_class[cls] += 1
            torchvision.io.write_jpeg(image, os.path.join(RETRIEVAL_IMAGES_DIR, 'mapillary.eval', f"{cls}---{d[cls]}___{picture}"))

    print('done!')


def main():
    # clean_ds()
    # create_german_ds()
    # create_unknown_ds()
    create_mapillary_ds()


if __name__ == '__main__':
    main()
