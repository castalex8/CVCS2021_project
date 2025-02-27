from typing import Tuple, List

import numpy as np
import torchvision

from signs.road_signs.German.dataset.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetTriplet(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        if not self.train:
            # Create testing dataset of triplets: [element, element in the same class, element in a different class]
            # For each element store its Path
            self.triplets = [[] for _ in range(len(self.img_labels))]
            for i in range(len(self.img_labels)):
                class_id = self.img_labels.iloc[i].ClassId
                self.triplets[i].append(self.img_labels.iloc[i])
                self.triplets[i].append(self.img_classes[class_id][np.random.randint(0, len(self.img_classes[class_id]))])
                negative = None
                negative_label = class_id
                while negative_label == class_id:
                    negative_label = np.random.randint(0, len(self.img_classes))
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                self.triplets[i].append(negative)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, index: int):
        # Creating during the training random triplets
        if self.train:
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
        else:
            # Return the triplet previously created
            anchor, positive, negative = self.triplets[index]

        return (
            self.transform(self.read_image(anchor)),
            self.transform(self.read_image(positive)),
            self.transform(self.read_image(negative))
        ), []
