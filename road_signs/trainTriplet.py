import os
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance
from torch.optim import Adam
from torch.utils.data import DataLoader
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets.GermanTrafficSignDatasetAbs import get_classes
from road_signs.datasets.GermanTrafficSignDatasetTriplet import GermanTrafficSignDatasetTriplet
from road_signs.train.Tripet import *
from road_signs.utils.Const import *


if __name__ == '__main__':
    classes = get_classes(is_local=LOCAL)
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = TripletNet().double()
    criterion = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance())
    classes = get_classes(is_local=LOCAL)
    optimizer = Adam(model.parameters(), lr=INIT_LR)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion.to(device)

    train(model, NUM_EPOCHS, optimizer, criterion, train_loader, device)
    torch.save(model.state_dict(), os.path.join('weights', 'weights/TripletWeights/3layer128deep.pth'))
    test(model, classes, test_loader, device)
