import os
from skimage import io, exposure, transform
import numpy as np
from road_signs.datasets.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs


class GermanTrafficSignDatasetTriplet(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None, is_local=True):
        super().__init__(train, trans, is_local)
        self.img_classes = [[]] * len(self.img_labels)
        for val in self.img_labels.values:
            self.img_classes[val[-2]].append(val[-1])

    def __len__(self):
        return len(self.img_labels)

    def format_image(self, item_path):
        img_path = os.path.join(self.base_dir, item_path)
        image = io.imread(img_path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        image.astype(np.double)
        return image

    def __getitem__(self, index):
        anchor = self.img_labels.iloc[index]
        anchor_label = anchor.ClassId
        positive_index = index

        while positive_index == index:
            positive_index = np.random.randint(0, len(self.img_classes[anchor_label]))

        positive = self.img_classes[anchor_label][positive_index]

        negative_index = 0
        negative_label = anchor_label

        while negative_label == anchor_label:
            negative_label = np.random.randint(0, len(self.img_classes))
            negative_index = np.random.randint(0, len(self.img_classes[negative_label]))

        negative = self.img_classes[anchor_label][negative_index]

        return (
            self.transform(self.format_image(anchor.Path)),
            self.transform(self.format_image(positive)),
            self.transform(self.format_image(negative))
        )
