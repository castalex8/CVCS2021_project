import torch
from torch.utils.data import DataLoader

from signs.road_signs.German.dataset.GermanTrafficSignDatasetAbs import CLASSES as GERMAN_CLASSES
from signs.road_signs.German.dataset.GermanTrafficSignDatasetClass import GermanTrafficSignDatasetClass
from signs.road_signs.Mapillary.dataset.MapillaryDatasetAbs import CLASSES as MAPILLARY_CLASSES
from signs.road_signs.Mapillary.dataset.MapillaryDatasetClass import MapillaryClass
from signs.road_signs.Unknown.dataset.UnknownDatasetAbs import CLASSES as UNKNOWN_CLASSES
from signs.road_signs.Unknown.dataset.UnknownDatasetClass import UnknownClass
from signs.road_signs.cnn.RoadSignNet import RoadSignNet
from signs.road_signs.train.Classification import test_pedestrian
from signs.road_signs.utils.Const import BS


def evaluate_mapillary_class(weigths: str):
    test_loader = DataLoader(MapillaryClass(train=False), batch_size=BS, shuffle=True)
    device_type = torch.device('cpu')
    model = RoadSignNet(classes=len(MAPILLARY_CLASSES))
    model.load_state_dict(torch.load(f'weigths/{weigths}', map_location=torch.device(device_type)))
    f = open(f'results_{weigths}.txt', 'w')
    pedestrian_label = 'information--pedestrians-crossing--g1'

    test_pedestrian(model, MAPILLARY_CLASSES, test_loader, device_type, pedestrian_label, f)


def evaluate_unknown_class(weigths: str):
    test_loader = DataLoader(UnknownClass(train=False), batch_size=BS, shuffle=True)
    device_type = torch.device('cpu')
    model = RoadSignNet(classes=len(UNKNOWN_CLASSES))
    model.load_state_dict(torch.load(f'weigths/{weigths}', map_location=torch.device(device_type)))
    f = open(f'results_{weigths}.txt', 'w')
    pedestrian_label = 'crosswalk'

    test_pedestrian(model, UNKNOWN_CLASSES, test_loader, device_type, pedestrian_label, f)


def evaluate_german_class(weigths: str):
    test_loader = DataLoader(GermanTrafficSignDatasetClass(train=False), batch_size=BS, shuffle=True)
    device_type = torch.device('cpu')
    model = RoadSignNet(classes=len(GERMAN_CLASSES)).double()
    model.load_state_dict(torch.load(f'weigths/{weigths}', map_location=torch.device(device_type)))
    f = open(f'results_{weigths}.txt', 'w')
    pedestrian_label = 'Pedestrians'

    test_pedestrian(model, GERMAN_CLASSES, test_loader, device_type, pedestrian_label, f)


if __name__ == '__main__':
    evaluate_mapillary_class('0010.pth')
    # evaluate_unknown_class('0020.pth')
    # evaluate_german_class('0004.pth')
