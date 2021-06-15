import numpy as np
from road_signs.datasets.GermanTrafficSignDatasetAbs import get_classes
from road_signs.datasets.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetSiamese(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, transform=None, is_local=True):
        super().__init__(train, transform, is_local)
        if not self.train:
            positive_pairs = [[] for i in range(len(self.img_labels) // 2)]
            negative_pairs = [[] for i in range(len(self.img_labels) // 2)]
            for i in range(len(positive_pairs)):
                # Match
                pos_img = self.img_labels.iloc[i]
                positive_pairs[i].append(pos_img.Path)
                positive_pairs[i].append(self.img_classes[pos_img.ClassId][np.random.randint(0, len(self.img_classes[pos_img.ClassId]))])
                positive_pairs[i].append(1)

            for i in range(len(negative_pairs)):
                # No match
                pos_img = self.img_labels.iloc[i]
                negative_pairs[i].append(pos_img.Path)
                negative_label = pos_img.ClassId
                negative = None
                while negative_label == pos_img.ClassId:
                    negative_label = np.random.randint(0, len(self.img_classes))
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                negative_pairs[i].append(negative)
                negative_pairs[i].append(0)

            self.pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1 = self.img_labels.iloc[index]
            img2 = None
            if target == 1:
                # Match
                img2 = img1.Path
                while img2 == img1.Path:
                    img2 = self.img_classes[img1.ClassId][np.random.randint(0, len(self.img_classes[img1.ClassId]))]
            else:
                # No match
                label2 = img1.ClassId
                while label2 == img1.ClassId:
                    label2 = np.random.randint(0, len(get_classes()))
                    img2 = self.img_classes[label2][np.random.randint(0, len(self.img_classes[label2]))]

            img1 = img1.Path
        else:
            img1, img2, target = self.pairs[index]

        return (self.transform(self.format_image(img1)), self.transform(self.format_image(img2))), target
