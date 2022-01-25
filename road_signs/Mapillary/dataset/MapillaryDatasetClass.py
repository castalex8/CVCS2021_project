import torchvision.io

from road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES, MapillaryDatasetAbs


class MapillaryClass(MapillaryDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> torchvision.io.image:
        item = self.labels[index]
        return self.transform(self.read_image(item)), CLASSES.index(item['label'])
