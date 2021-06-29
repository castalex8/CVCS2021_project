import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets.GermanTrafficSignDatasetTriplet import GermanTrafficSignDatasetTriplet
from road_signs.utils.Const import *
from road_signs.train.Tripet import train_epoch, test_epoch
from road_signs.train.fit import fit


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)

    model = TripletNet().double()
    loss_fn = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(), margin=MARGIN)
    cuda = torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, NUM_EPOCHS, cuda, train_epoch, test_epoch)
    torch.save(model.state_dict(), '../weights/TripletWeights/FitTriplet3layer128deep10epochs.pth')
