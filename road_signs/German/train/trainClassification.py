import os

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from road_signs.German.dataset.GermanTrafficSignDatasetAbs import CLASSES
from road_signs.German.dataset.GermanTrafficSignDatasetClass import GermanTrafficSignDatasetClass
from road_signs.cnn.RoadSignNet import RoadSignNet
from road_signs.train.Classification import *
from road_signs.utils.Const import *


if __name__ == '__main__':
    # load the label names
    train_loader = DataLoader(GermanTrafficSignDatasetClass(train=True), batch_size=BS, shuffle=True, pin_memory=bool(os.getenv('USE_LAB')))
    test_loader = DataLoader(GermanTrafficSignDatasetClass(train=False), batch_size=BS, shuffle=True, pin_memory=bool(os.getenv('USE_LAB')))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # defining the model
    model = RoadSignNet().double()
    model.to(device)

    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    # defining the loss function
    criterion = CrossEntropyLoss()

    with open('class.txt', 'a') as fout:
        train(model, NUM_EPOCHS, optimizer, criterion, train_loader, device, fout)
        torch.save(model.state_dict(), '../../weights/German/class.pth')
        test(model, CLASSES, test_loader, device, fout)
