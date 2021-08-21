from collections import defaultdict
from copy import deepcopy

from road_signs.Mapillary.dataset.MapillaryDatasetAbs import MapillaryDatasetAbs


class MapillaryDatasetRetr(MapillaryDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)
        self.img_classes = defaultdict(list)
        for label in self.labels:
            if label['label'] != 'other-sign':
                self.img_classes[label['label']].append(label)

        # Take only big classes
        tmp_img_classes = deepcopy(self.img_classes)
        for key, val in tmp_img_classes.items():
            if len(val) < 5:
                print(key, len(val))

    def __getitem__(self, index):
        raise NotImplemented
