import torch
import torchvision
from torch import nn, optim
from torch.nn import PairwiseDistance
from torch.optim import lr_scheduler
from road_signs.cnn.RoadSignNetFC import get_road_sign_fc
from road_signs.datasets.GermanTrafficSignDatasetAbs import is_local
from road_signs.train.Tripet import train_epoch, test_epoch
from road_signs.train.fit import fit
from road_signs.utils.Const import *
from torch.utils.data import DataLoader
from road_signs.datasets.GermanTrafficSignDatasetTripletOnline import GermanTrafficSignDatasetTriplet


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_conv.fc = get_road_sign_fc(model_conv.fc.in_features)
    # model_conv.fc = get_road_sign_fc(model_conv.fc.in_features)
    model_conv = model_conv.double()
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True), batch_size=BS, shuffle=True, pin_memory=not is_local())
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False), batch_size=BS, shuffle=True, pin_memory=not is_local())
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(), margin=MARGIN)
    print(device)
    model_conv.to(device)
    loss_fn.to(device)

    # Observe that only parameters of final layer are being optimized as opposed to before.
    optim = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

    fit(train_loader, test_loader, model_conv, loss_fn, optim, scheduler, NUM_EPOCHS, cuda, train_epoch, test_epoch)
    torch.save(model_conv.state_dict(), 'weights/TripletWeights/preTrainedFitTriplet3layer256deep.pth')
