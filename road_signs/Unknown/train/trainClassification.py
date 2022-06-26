from torch.utils.data import DataLoader

from road_signs.Unknown.dataset.UnknownDatasetAbs import CLASSES
from road_signs.Unknown.dataset.UnknownDatasetClass import UnknownClass
from road_signs.utils.Const import BS, use_lab
from road_signs.utils.train import train_classification


if __name__ == '__main__':
    # load the label names
    train_loader = DataLoader(UnknownClass(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(UnknownClass(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())
    train_classification(train_loader, test_loader, classes=CLASSES, is_double=False)
