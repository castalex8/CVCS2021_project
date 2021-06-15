from torch import optim
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

    model = SiameseNet().double()
    loss_fn = ContrastiveLoss(margin=MARGIN)
    cuda = torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, NUM_EPOCHS, cuda, train_epoch, test_epoch)
    torch.save(model.state_dict(), '../weights/SiameseWeights/FitSiamese3layer128deep.pth')
