import numpy as np
from road_signs.datasets.GermanTrafficSignDatasetRetr import GermanTrafficSignDatasetRetr


class GermanTrafficSignDatasetTriplet(GermanTrafficSignDatasetRetr):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        if not self.train:
            self.triplets = [[] for i in range(len(self.img_labels))]
            for i in range(len(self.img_labels)):
                class_id = self.img_labels.iloc[i].ClassId
                self.triplets[i].append(self.img_labels.iloc[i].Path)
                self.triplets[i].append(self.img_classes[class_id][np.random.randint(0, len(self.img_classes[class_id]))])
                negative = None
                negative_label = class_id
                while negative_label == class_id:
                    negative_label = np.random.randint(0, len(self.img_classes))
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                self.triplets[i].append(negative)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
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
            anchor = anchor.Path
        else:
            anchor, positive, negative = self.triplets[index]

        return (
            self.transform(self.read_image(anchor)),
            self.transform(self.read_image(positive)),
            self.transform(self.read_image(negative))
        ), []
