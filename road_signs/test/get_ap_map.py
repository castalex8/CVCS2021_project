import os
from typing import List

from torchvision.io import read_image

from road_signs.datasets_utils import get_dataset
from road_signs.retrieval import retrieve_top_n_results

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


def main():

    aps = []

    for f in os.listdir(SAMPLE_IMAGES_FOLDER):
        img = read_image(os.path.join(SAMPLE_IMAGES_FOLDER, f))
        results = retrieve_top_n_results(img, NET, SCOPE)

        aps.append(get_ap([ds['get_label'](res[1]) for res in results]))

    print(f'AP: {aps}')
    print(f'MAP: {sum(aps) / len(aps)}')


if __name__ == '__main__':
    main()
