from torch.utils.data import DataLoader

from signs.road_signs.Unknown.dataset.UnknownDatasetTriplet import UnknownDatasetTriplet
from signs.road_signs.utils.Const import BS, use_lab
from signs.road_signs.utils.train import train_triplet


if __name__ == '__main__':
    train_loader = DataLoader(UnknownDatasetTriplet(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(UnknownDatasetTriplet(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())

    train_triplet(train_loader, test_loader)
