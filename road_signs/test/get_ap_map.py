import os
from typing import List

from road_signs.German.dataset.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs, CLASSES as GERMAN_CLASSES
from road_signs.Mapillary.dataset.MapillaryDatasetAbs import MapillaryDatasetAbs
from road_signs.Unknown.dataset.UnknownDatasetAbs import UnknownDatasetAbs
from road_signs.datasets_utils import get_dataset
from road_signs.siamese_retrieval import retrieve_siamese_top_n_results
from road_signs.triplet_retrieval import retrieve_triplet_top_n_results

ds = get_dataset()
SAMPLE_IMAGES_FOLDER = os.getenv('SAMPLE_IMAGES_FOLDER')
DATASET = os.getenv('DATASET')
SCOPE = 10
NET = 'siamese'


def get_ap(labels: List[str]) -> float:
    crosswalk_label = ds['get_crosswalk_label']()
    ap = 0
    count = 0

    for i, l in enumerate(labels):
        if l == crosswalk_label:
            count += 1

        ap += count / (i + 1)

    return ap / len(labels)


def evaluate_mapillary_siamese():
    aps = {
        10: [],
        20: []
    }

    map_ds = MapillaryDatasetAbs(train=False)
    for annotation in map_ds.labels:
        if annotation['label'] == 'information--pedestrians-crossing--g1':
            img = map_ds.read_image(annotation)
            results = retrieve_siamese_top_n_results(img, 20)
            aps[10].append(get_ap([os.path.basename(res[1].split('---')[0]) for res in results[:10]]))
            aps[20].append(get_ap([os.path.basename(res[1].split('---')[0]) for res in results]))

    print(f'AP: {aps}')
    print(f'Mapillary Siamese MAP 10: {sum(aps[10]) / len(aps[10])}')
    print(f'Mapillary Siamese MAP 20: {sum(aps[20]) / len(aps[20])}')


def evaluate_mapillary_triplet():
    aps = []

    map_ds = MapillaryDatasetAbs(train=False)
    for annotation in map_ds.labels:
        if annotation['label'] == 'information--pedestrians-crossing--g1':
            img = map_ds.read_image(annotation)
            results = retrieve_triplet_top_n_results(img, 10)
            aps.append(get_ap([os.path.basename(res[1].split('---')[0]) for res in results]))

    print(f'Mapillary Triplet AP: {aps}')
    print(f'Mapillary Triplet MAP: {sum(aps) / len(aps)}')


def evaluate_unknown_siamese():
    aps = []

    map_ds = UnknownDatasetAbs(train=False)
    for annotation in map_ds.labels:
        if annotation['label'] == 'crosswalk':
            img = map_ds.read_image(annotation)
            results = retrieve_siamese_top_n_results(img, 10)
            aps.append(get_ap([os.path.basename(res[1].split('-')[0]) for res in results]))

    print(f'Unknown Siamese AP: {aps}')
    print(f'Unknown Siamese MAP: {sum(aps) / len(aps)}')


def evaluate_unknown_triplet():
    aps = []

    map_ds = UnknownDatasetAbs(train=False)
    for annotation in map_ds.labels:
        if annotation['label'] == 'crosswalk':
            img = map_ds.read_image(annotation)
            results = retrieve_triplet_top_n_results(img, 10)
            aps.append(get_ap([os.path.basename(res[1].split('-')[0]) for res in results]))

    print(f'Unknown Triplet AP: {aps}')
    print(f'Unknown Triplet MAP: {sum(aps) / len(aps)}')


def evaluate_german_siamese():
    aps = {
        10: [],
        20: []
    }

    map_ds = GermanTrafficSignDatasetAbs(train=False)

    for i in range(len(map_ds.img_labels)):
        annotation = map_ds.img_labels.iloc[i]
        if GERMAN_CLASSES[annotation.ClassId] == 'Pedestrians':
            img = map_ds.read_image(annotation)
            results = retrieve_siamese_top_n_results(img, 20)
            aps[10].append(get_ap([GERMAN_CLASSES[int(os.path.basename(res[1]).split('_')[0])] for res in results[:10]]))
            aps[20].append(get_ap([GERMAN_CLASSES[int(os.path.basename(res[1]).split('_')[0])] for res in results]))

    print(f'German Siamese AP: {aps}')
    print(f'German Siamese MAP 10: {sum(aps[10]) / len(aps[10])}')
    print(f'German Siamese MAP 20: {sum(aps[20]) / len(aps[20])}')


def evaluate_german_triplet():
    aps = {
        10: [],
        20: []
    }

    map_ds = GermanTrafficSignDatasetAbs(train=False)
    for i in range(len(map_ds.img_labels)):
        annotation = map_ds.img_labels.iloc[i]
        if GERMAN_CLASSES[annotation.ClassId] == 'Pedestrians':
            img = map_ds.read_image(annotation)
            results = retrieve_triplet_top_n_results(img, 20)
            aps[10].append(get_ap([GERMAN_CLASSES[int(os.path.basename(res[1]).split('_')[0])] for res in results[:10]]))
            aps[20].append(get_ap([GERMAN_CLASSES[int(os.path.basename(res[1]).split('_')[0])] for res in results]))

    print(f'German Triplet AP: {aps}')
    print(f'German Triplet MAP: {sum(aps[10]) / len(aps[10])}')
    print(f'German Triplet MAP: {sum(aps[20]) / len(aps[20])}')


def main():
    evaluate_mapillary_siamese()
    evaluate_mapillary_triplet()

    evaluate_unknown_siamese()
    evaluate_unknown_triplet()

    evaluate_german_siamese()
    evaluate_german_triplet()


if __name__ == '__main__':
    main()
