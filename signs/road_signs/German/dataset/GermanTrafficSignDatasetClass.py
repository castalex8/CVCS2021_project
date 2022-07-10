import os
from typing import Tuple

import torchvision

from signs.road_signs.German.dataset.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs


class GermanTrafficSignDatasetClass(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, index: int) -> torchvision.io.image:
        item = self.img_labels.iloc[index]
        return self.transform(self.read_image(item)), item.ClassId
