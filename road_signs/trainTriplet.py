from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets.GermanTrafficSignDatasetTriplet import GermanTrafficSignDatasetTriplet
from road_signs.utils.Const import *
from road_signs.train.Tripet import *
from road_signs.train.fit import fit


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)

    margin = 1.
    model = TripletNet().double()
    loss_fn = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(), margin=margin)
    lr = 1e-3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10
    log_interval = 100

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, str(device) == 'cuda', log_interval)
    torch.save(model.state_dict(), '../weights/TripletWeights/FitTriplet3layer128deep10epochs.pth')
