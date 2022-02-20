from torch.utils.data import DataLoader

from road_signs.Unknown.dataset.UnknownDatasetSiamese import UnknownSiamese
from road_signs.utils.Const import BS, use_lab
from road_signs.utils.train import train_siamese


if __name__ == '__main__':
    train_loader = DataLoader(UnknownSiamese(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(UnknownSiamese(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())

    train_siamese(train_loader, test_loader)
