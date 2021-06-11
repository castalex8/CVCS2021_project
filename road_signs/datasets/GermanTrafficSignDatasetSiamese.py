import numpy as np

from road_signs.datasets.GermanTrafficSignDatasetAbs import get_classes
from road_signs.datasets.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetSiamese(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, transform=None, is_local=True):
        super().__init__(train, transform, is_local)

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1 = self.img_labels.iloc[index]

        if target == 0:
            label2 = img1.ClassId
            while label2 == img1.ClassId:
                label2 = np.random.randint(0, len(get_classes()))
                img2 = self.img_classes[label2][np.random.randint(0, len(self.img_classes[label2]))]
        else:
            img2 = img1.Path
            while img2 == img1.Path:
                img2 = self.img_classes[img1.ClassId][np.random.randint(0, len(self.img_classes[img1.ClassId]))]

        return self.transform(self.format_image(img1.Path)), self.transform(self.format_image(img2)), img1.ClassId
