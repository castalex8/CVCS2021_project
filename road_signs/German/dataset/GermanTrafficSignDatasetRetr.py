from road_signs.German.dataset.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs, CLASSES


class GermanTrafficSignDatasetRetr(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        self.img_classes = [[] for i in range(len(CLASSES))]
        for val in self.img_labels.values:
            self.img_classes[val[-2]].append(val[-1])

    def __getitem__(self, index):
        raise NotImplemented
