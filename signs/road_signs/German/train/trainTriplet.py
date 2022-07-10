from torch.utils.data import DataLoader

from signs.road_signs.German.dataset.GermanTrafficSignDatasetTriplet import GermanTrafficSignDatasetTriplet
from signs.road_signs.utils.Const import BS, use_lab
from signs.road_signs.utils.train import train_triplet


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())

    train_triplet(train_loader, test_loader, is_double=True)
