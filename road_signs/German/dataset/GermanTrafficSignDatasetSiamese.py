from typing import Tuple

import numpy as np
import torchvision

from road_signs.German.dataset.GermanTrafficSignDatasetAbs import CLASSES
from road_signs.German.dataset.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetSiamese(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, transform=None):
        super().__init__(train, transform)
        # The testing dataset is created once during the class initialization
        if not self.train:
            # Split the dataset in half, with positive and negative pairs, in order to create
            # a dataset test with the same length of the dataset training set.
            positive_pairs = [[] for _ in range(len(self.img_labels) // 2)]
            negative_pairs = [[] for _ in range(len(self.img_labels) // 2)]
            for i in range(len(positive_pairs)):
                # Match - take 2 elements of the same class. It could be also the same image
                pos_img = self.img_labels.iloc[i]
                positive_pairs[i].append(pos_img.Path)
                positive_pairs[i].append(self.img_classes[pos_img.ClassId][np.random.randint(0, len(self.img_classes[pos_img.ClassId]))])
                positive_pairs[i].append(1)

            for i in range(len(negative_pairs)):
                # No match - take 2 elements from different classes
                pos_img = self.img_labels.iloc[i]
                negative_pairs[i].append(pos_img.Path)
                negative_label = pos_img.ClassId
                negative = None
                while negative_label == pos_img.ClassId:
                    negative_label = np.random.randint(0, len(self.img_classes))
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                negative_pairs[i].append(negative)
                negative_pairs[i].append(0)

            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index: int) -> Tuple[tuple[torchvision.io.image, torchvision.io.image], bool]:
        # Create during the training random couple of negative and positive items
        if self.train:
            # Take randomly a negative or a positive pair
            target = np.random.randint(0, 2)
            img1 = self.img_labels.iloc[index]
            img2 = None
            if target == 1:
                # Match - choose a different image from the same class
                img2 = img1.Path
                while img2 == img1.Path:
                    img2 = self.img_classes[img1.ClassId][np.random.randint(0, len(self.img_classes[img1.ClassId]))]
            else:
                # No match - choose an image from a different class
                label2 = img1.ClassId
                while label2 == img1.ClassId:
                    label2 = np.random.randint(0, len(CLASSES))
                    img2 = self.img_classes[label2][np.random.randint(0, len(self.img_classes[label2]))]

            img1 = img1.Path
        else:
            # Return the pairs previously created
            img1, img2, target = self.test_pairs[index]

        return (self.transform(self.read_image(img1)), self.transform(self.read_image(img2))), target
