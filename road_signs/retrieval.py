from collections import defaultdict
from typing import List

import torchvision

from road_signs.datasets_utils import get_dataset
from road_signs.siamese_retrieval import retrieve_siamese_top_n_results
from road_signs.triplet_retrieval import retrieve_triplet_top_n_results


def retrieve_top_n_results(img: torchvision.io.image, net: str = 'siamese', max_results: int = 10) -> List[dict]:
    return retrieve_siamese_top_n_results(img, max_results) if net == 'siamese' \
        else retrieve_triplet_top_n_results(img, max_results)


def retrieve_most_similar_result(img: torchvision.io.image, net: str = 'siamese') -> List[dict]:
    return retrieve_siamese_top_n_results(img, 1) if net == 'siamese' else retrieve_triplet_top_n_results(img, 1)


def retrieve_most_similar_label(img: torchvision.io.image, net: str = 'siamese') -> int:
    ds = get_dataset()
    sort_losses = retrieve_siamese_top_n_results(img, 10) if net == 'siamese' else retrieve_triplet_top_n_results(img, 10)
    label_occurrences = defaultdict(int)

    for res in sort_losses:
        label_occurrences[ds['get_label'](res[1])] += 1

    return max(label_occurrences, key=label_occurrences.get)
