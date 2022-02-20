from torch.utils.data import DataLoader

from road_signs.Mapillary.dataset.MapillaryDatasetTriplet import MapillaryDatasetTriplet
from road_signs.utils.Const import BS, use_lab
from road_signs.utils.train import train_triplet


if __name__ == '__main__':
    train_loader = DataLoader(MapillaryDatasetTriplet(train=True), batch_size=BS, shuffle=True, pin_memory=use_lab())
    test_loader = DataLoader(MapillaryDatasetTriplet(train=False), batch_size=BS, shuffle=True, pin_memory=use_lab())

    train_triplet(train_loader, test_loader)
