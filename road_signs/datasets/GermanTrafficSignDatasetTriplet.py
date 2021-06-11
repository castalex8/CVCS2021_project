import numpy as np
from road_signs.datasets.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetTriplet(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, trans=None, is_local=True):
        super().__init__(train, trans, is_local)

    def __len__(self):
        return len(self.img_labels)

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

        negative = self.img_classes[negative_label][negative_index]

        return (
            self.transform(self.format_image(anchor.Path)),
            self.transform(self.format_image(positive)),
            self.transform(self.format_image(negative))
        )
