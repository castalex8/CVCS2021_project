import numpy as np
from road_signs.Mapillary.dataset.MapillaryDatasetRetr import MapillaryDatasetRetr


class MapillaryDatasetTriplet(MapillaryDatasetRetr):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        if not self.train:
            self.triplets = [[] for i in range(len(self.labels))]
            for i in range(len(self.labels)):
                anchor = self.labels[i]
                self.triplets[i].append(anchor)

                positive = anchor
                while positive['path'] == anchor['path']:
                    positive = self.img_classes[anchor['label']][np.random.randint(0, len(self.img_classes[anchor['label']]))]
                self.triplets[i].append(positive)

                negative = anchor
                while negative['label'] == anchor['label']:
                    negative_label = list(self.img_classes.keys())[np.random.randint(0, len(self.img_classes))]
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                self.triplets[i].append(negative)

    def __getitem__(self, index):
        if self.train:
            anchor = self.labels[index]
            positive = anchor

            while positive['path'] == anchor['path']:
                positive = self.img_classes[anchor['label']][np.random.randint(0, len(self.img_classes[anchor['label']]))]

            negative = anchor
            while negative['label'] == anchor['label']:
                negative_label = list(self.img_classes.keys())[np.random.randint(0, len(self.img_classes))]
                negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]
        else:
            anchor, positive, negative = self.triplets[index]

        return (
            self.transform(self.read_image(anchor)),
            self.transform(self.read_image(positive)),
            self.transform(self.read_image(negative))
        ), []
