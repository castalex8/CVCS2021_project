from torch.utils.data import DataLoader

from road_signs.German.dataset.GermanTrafficSignDatasetSiamese import GermanTrafficSignDatasetSiamese
from road_signs.utils.Const import BS, use_lab
from road_signs.utils.train import train_siamese


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())

    train_siamese(train_loader, test_loader, is_double=True)
