from typing import Tuple

import torchvision

from road_signs.Unknown.dataset.UnknownDatasetRetr import UnknownDatasetRetr
from road_signs.utils.create_ds import create_test_pairs, create_online_training_couple


class UnknownSiamese(UnknownDatasetRetr):
    def __init__(self, train=True, transform=None):
        super().__init__(train, transform)
        if not self.train:
            self.test_pairs = create_test_pairs(self.labels, self.img_classes)

    def __getitem__(self, index: int) -> Tuple[tuple[torchvision.io.image, torchvision.io.image], bool]:
        if self.train:
            img1, img2, target = create_online_training_couple(self.labels[index], self.img_classes)
        else:
            img1, img2, target = self.test_pairs[index]

        return (self.transform(self.read_image(img1)), self.transform(self.read_image(img2))), target
