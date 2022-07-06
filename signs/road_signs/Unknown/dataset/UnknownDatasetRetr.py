from collections import defaultdict
from road_signs.Unknown.dataset.UnknownDatasetAbs import UnknownDatasetAbs


class UnknownDatasetRetr(UnknownDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        self.img_classes = defaultdict(list)
        for label in self.labels:
            self.img_classes[label['label']].append(label)

    def __getitem__(self, index: int):
        raise NotImplemented
