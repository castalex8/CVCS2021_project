import os

import torch
import torchvision
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from signs.road_signs.Mapillary.dataset.MapillaryDatasetSiamese import MapillarySiamese
from signs.road_signs.cnn.RoadSignNetFC import get_road_sign_fc
from signs.road_signs.loss.ConstrastiveLoss import ContrastiveLoss
from signs.road_signs.train.Siamese import train_epoch, test_epoch
from signs.road_signs.utils.train import fit
from signs.road_signs.utils.Const import BS, MARGIN, INIT_LR, STEP_SIZE, GAMMA, NUM_EPOCHS, MOMENTUM

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    model_conv.fc = get_road_sign_fc()

    train_loader = DataLoader(MapillarySiamese(train=True), batch_size=BS, shuffle=True, pin_memory=bool(os.getenv('USE_LAB')))
    test_loader = DataLoader(MapillarySiamese(train=False), batch_size=BS, shuffle=True, pin_memory=bool(os.getenv('USE_LAB')))
    loss_fn = ContrastiveLoss(margin=MARGIN)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_conv.to(device)
    loss_fn.to(device)

    # Observe that only parameters of final layer are being optimized as opposed to before.
    optim = optim.SGD(model_conv.fc.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)

    fit(train_loader, test_loader, model_conv, loss_fn, optim, NUM_EPOCHS, cuda, train_epoch, test_epoch, 'pretrained_siamese.txt', scheduler)
    torch.save(model_conv.state_dict(), '../../weights/Mapillary/preTrainedSiamese.pth')
