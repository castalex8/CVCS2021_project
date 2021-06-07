from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from road_signs.cnn.RoadSignNet import RoadSignNet
from road_signs.datasets.GermanTrafficSignDataset import GermanTrafficSignDataset, get_classes
from road_signs.utils.train_with_loaders import *


# initialize the number of epochs to train for, base learning rate, and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
MOMENTUM = 0.9
BS = 64


if __name__ == '__main__':
    # load the label names
    classes = get_classes()
    train_loader = DataLoader(GermanTrafficSignDataset(train=True), batch_size=BS, shuffle=True, num_workers=8)
    test_loader = DataLoader(GermanTrafficSignDataset(train=False), batch_size=BS, shuffle=True, num_workers=8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # defining the model
    model = RoadSignNet().double()
    model.to(device)

    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    # defining the loss function
    criterion = CrossEntropyLoss()

    train_with_dataloader(model, NUM_EPOCHS, optimizer, criterion, train_loader)
    torch.save(model.state_dict(), 'weights/weights.pth')
    test_with_dataloader(model, classes, test_loader)

    # for t in range(NUM_EPOCHS):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train_loop(train_loader, model, criterion, optimizer)
    #     test_loop(test_loader, model, criterion)
