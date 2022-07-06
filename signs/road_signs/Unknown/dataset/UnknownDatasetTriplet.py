from typing import Tuple, List

import torchvision

from road_signs.Unknown.dataset.UnknownDatasetRetr import UnknownDatasetRetr
from road_signs.utils.create_ds import create_test_triplets, create_online_training_triplet


class UnknownDatasetTriplet(UnknownDatasetRetr):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        if not self.train:
            self.triplets = create_test_triplets(self.labels, self.img_classes)

    def __getitem__(self, index: int) -> Tuple[tuple[torchvision.io.image, torchvision.io.image, torchvision.io.image], List]:
        if self.train:
            anchor, positive, negative = create_online_training_triplet(self.labels[index], self.img_classes)
        else:
            anchor, positive, negative = self.triplets[index]

        return (
            self.transform(self.read_image(anchor)),
            self.transform(self.read_image(positive)),
            self.transform(self.read_image(negative))
        ), []
