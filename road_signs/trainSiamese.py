from torch import optim
from torch.nn import CosineSimilarity
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets.GermanTrafficSignDatasetSiamese import GermanTrafficSignDatasetSiamese
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.utils.Const import *
from road_signs.train.fit import fit
from road_signs.train.Siamese import *


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)

    margin = 1.
    model = SiameseNet().double()
    loss_fn = ContrastiveLoss(margin=margin)
    lr = 1e-3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100

    fit(
        train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs,
        str(device) == 'cuda', train_epoch, test_epoch
    )
    torch.save(model.state_dict(), '../weights/SiameseWeights/FitSiamese3layer128deep.pth')
