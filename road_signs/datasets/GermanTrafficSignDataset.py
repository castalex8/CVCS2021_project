import os
from skimage import io, exposure, transform
import numpy as np
from road_signs.datasets.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs


class GermanTrafficSignDataset(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        item = self.img_labels.iloc[index]
        img_path = os.path.join(self.base_dir, item.Path)
        image = io.imread(img_path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        image.astype(np.double)
        label = item.ClassId

        return self.transform(image), label
