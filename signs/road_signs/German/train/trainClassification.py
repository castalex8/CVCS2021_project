from torch.utils.data import DataLoader

from road_signs.German.dataset.GermanTrafficSignDatasetClass import GermanTrafficSignDatasetClass
from road_signs.utils.Const import BS, use_lab
from road_signs.utils.train import train_classification


if __name__ == '__main__':
    # load the label names
    train_loader = DataLoader(GermanTrafficSignDatasetClass(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(GermanTrafficSignDatasetClass(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())
    train_classification(train_loader, test_loader, classes=43, is_double=True)
