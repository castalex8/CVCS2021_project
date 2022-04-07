import json
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES

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
    exists_or_create('german')
    print('Creating german dataset... ', end='')
    for folder in range(43):
        base_path = os.path.join(GERMAN_PATH, 'Train', str(folder))
        for i, picture in enumerate(os.listdir(base_path)):
            if '.png' not in picture:
                continue

            if i >= NUMBER_PICTURES:
                break

            shutil.copy(
                os.path.join(base_path, picture),
                os.path.join(RETRIEVAL_IMAGES_DIR, 'german', f'{i}__{picture}')
            )
    print('done!')


def create_unknown_ds():
    # Create Unknown retrieval set
    exists_or_create('unknown')
    print('Creating unknown dataset... ', end='')
    d = defaultdict(int)
    base_path = os.path.join(UNKNOWN_PATH, 'images')
    for picture in os.listdir(base_path):
        if len(d) > 0 and all([count > NUMBER_PICTURES for count in d.values()]):
            break

        base_name = picture[:-4]
        ann_doc = ET.parse(os.path.join(UNKNOWN_PATH, 'annotations', f"{base_name}.xml")).getroot()
        label = ann_doc.find("./object/name").text
        if d[label] >= NUMBER_PICTURES:
            continue

        d[label] += 1
        shutil.copy(
            os.path.join(base_path, picture),
            os.path.join(RETRIEVAL_IMAGES_DIR, 'unknown', f"{label}-{d[label]}__{picture}")
        )
    print('done!')


def create_mapillary_ds():
    # Create Mapillary retrieval set
    exists_or_create('mapillary')
    print('Creating unknown mapillary... ', end='')
    d = defaultdict(int)
    base_path = os.path.join(MAPILLARY_PATH, 'images.train.0')
    for picture in os.listdir(base_path):
        if len(d) == len(MAPILLARY_PATH) and all([count > NUMBER_PICTURES for count in d.values()]):
            break

        base_name = picture[:-4]
        classes = [obj['label'] for obj in json.load(
            open(os.path.join(MAPILLARY_PATH, 'annotations', 'annotations', f"{base_name}.json"))
        )['objects'] if obj['label'] in MAPILLARY_CLASSES]

        for cls in classes:
            if d[cls] >= NUMBER_PICTURES:
                continue

            d[cls] += 1
            shutil.copy(
                os.path.join(base_path, picture),
                os.path.join(RETRIEVAL_IMAGES_DIR, 'mapillary', f"{cls}---{d[cls]}___{picture}")
            )
    print('done!')


def main():
    clean_ds()
    create_german_ds()
    create_unknown_ds()
    create_mapillary_ds()


if __name__ == '__main__':
    main()
