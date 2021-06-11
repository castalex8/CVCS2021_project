import os
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.train.Siamese import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from road_signs.cnn.SiameseNet import SiameseNet
from road_signs.datasets.GermanTrafficSignDatasetAbs import get_classes
from road_signs.datasets.GermanTrafficSignDatasetSiamese import GermanTrafficSignDatasetSiamese
from road_signs.utils.Const import *


if __name__ == '__main__':
    train_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)

    criterion = ContrastiveLoss(MARGIN)
    model = SiameseNet().double()
    optimizer = Adam(model.parameters(), lr=INIT_LR)
    classes = get_classes(is_local=LOCAL)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion.to(device)

    train(model, NUM_EPOCHS, optimizer, criterion, train_loader, device)
    torch.save(model.state_dict(), os.path.join('weights', 'weights/SiamesetWeights/3layer128deep.pth'))
    test(model, classes, test_loader, device)
