from road_signs.Unknown.dataset.UnknownDatasetAbs import UnknownDatasetAbs, CLASSES


class UnknownClass(UnknownDatasetAbs):
    def __init__(self, train=True, trans=None):
        super().__init__(train, trans)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = self.labels[index]
        return self.transform(self.read_image(item)), CLASSES.index(item['label'])
