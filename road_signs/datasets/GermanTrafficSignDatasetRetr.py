from road_signs.datasets.GermanTrafficSignDatasetAbs import GermanTrafficSignDatasetAbs, get_classes


class GermanTrafficSignDatasetRetr(GermanTrafficSignDatasetAbs):
    def __init__(self, train=True, trans=None, is_local=True):
        super().__init__(train, trans, is_local)
        self.img_classes = [[] for i in range(len(get_classes()))]
        for val in self.img_labels.values:
            self.img_classes[val[-2]].append(val[-1])

    def __getitem__(self, index):
        raise NotImplemented
