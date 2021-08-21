import numpy as np
from road_signs.Mapillary.dataset.MapillaryDatasetRetr import MapillaryDatasetRetr


class MapillarySiamese(MapillaryDatasetRetr):
    def __init__(self, train=True, transform=None):
        super().__init__(train, transform)
        if not self.train:
            positive_pairs = [[] for i in range(len(self.labels) // 2)]
            negative_pairs = [[] for i in range(len(self.labels) - len(positive_pairs))]
            for i in range(len(positive_pairs)):
                # Match
                pos_img = self.labels[i]
                positive_pairs[i].append(pos_img)
                pos_img2 = pos_img
                while pos_img2['path'] == pos_img['path']:
                    pos_img2 = self.img_classes[pos_img['label']][np.random.randint(0, len(self.img_classes[pos_img['label']]))]

                positive_pairs[i].append(pos_img2)
                positive_pairs[i].append(1)

            for i in range(len(negative_pairs)):
                # No match
                pos_img = self.labels[i]
                negative_pairs[i].append(pos_img)
                negative_label = pos_img['label']
                negative = None
                while negative_label == pos_img['label']:
                    negative_label = list(self.img_classes.keys())[np.random.randint(0, len(self.img_classes))]
                    negative = self.img_classes[negative_label][np.random.randint(0, len(self.img_classes[negative_label]))]

                negative_pairs[i].append(negative)
                negative_pairs[i].append(0)

            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1 = self.labels[index]
            img2 = None
            if target == 1:
                # Match
                img2 = img1
                while img2['path'] == img1['path']:
                    img2 = self.img_classes[img1['label']][np.random.randint(0, len(self.img_classes[img1['label']]))]
            else:
                # No match
                label2 = img1['label']
                while label2 == img1['label']:
                    label2 = list(self.img_classes.keys())[np.random.randint(0, len(self.img_classes.keys()))]
                    img2 = self.img_classes[label2][np.random.randint(0, len(self.img_classes[label2]))]

        else:
            img1, img2, target = self.test_pairs[index]

        if img1['path'] == img2['path'] and img1['label'] == img2['label']:
            print('bad', img1, img2)

        return (self.transform(self.read_image(img1)), self.transform(self.read_image(img2))), target
