from road_signs.German.dataset.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs, CLASSES


class GermanTrafficSignDatasetRetr(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        self.img_classes = [[] for _ in range(len(CLASSES))]

        # Add to each class the relative images' path loading the complete dataset
        for i in range(len(self.img_labels)):
            self.img_classes[self.img_labels.iloc[i].ClassId].append(self.img_labels.iloc[i])

    def __getitem__(self, index: int):
        raise NotImplemented
