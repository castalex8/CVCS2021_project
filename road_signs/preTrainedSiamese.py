import torch
import torchvision
from torch import optim
from torch.optim import lr_scheduler
from road_signs.cnn.RoadSignNetFC import get_road_sign_fc
from road_signs.datasets.GermanTrafficSignDatasetSiameseOnline import GermanTrafficSignDatasetSiamese
from road_signs.loss.CostrastiveLoss import ContrastiveLoss
from road_signs.train.Siamese import train_epoch, test_epoch
from road_signs.train.fit import fit
from road_signs.utils.Const import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    model_conv.fc = get_road_sign_fc()

    model_conv = model_conv.double()
    train_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=True), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetSiamese(train=False), batch_size=BS, shuffle=True)
    loss_fn = ContrastiveLoss(margin=MARGIN)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_conv.to(device)
    loss_fn.to(device)

    # Observe that only parameters of final layer are being optimized as opposed to before.
    optim = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

    fit(train_loader, test_loader, model_conv, loss_fn, optim, scheduler, NUM_EPOCHS, cuda, train_epoch, test_epoch, 'pretrained_siamese.txt')
    torch.save(model_conv.state_dict(), 'weights/SiameseWeights/preTrainedFitSiamese3layer256deep.pth')
