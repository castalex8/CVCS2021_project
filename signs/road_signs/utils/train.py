import datetime
from typing import List, Callable

import torch
from torch import optim, nn
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance, CrossEntropyLoss
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader

from signs.road_signs.cnn.RoadSignNet import RoadSignNet
from signs.road_signs.cnn.SiameseNet import SiameseNet
from signs.road_signs.cnn.TripletNet import TripletNet
from signs.road_signs.loss.ConstrastiveLoss import ContrastiveLoss
from signs.road_signs.train.Triplet import train_epoch as triplet_train_epoch, test_epoch as triplet_test_epoch
from signs.road_signs.train.Siamese import train_epoch as siamese_train_epoch, test_epoch as siamese_test_epoch
from signs.road_signs.train.Classification import train as class_train, test as class_test
from signs.road_signs.utils.Const import MARGIN, GAMMA, NUM_EPOCHS, STEP_SIZE, INIT_LR, MOMENTUM


def fit(
    train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, loss_fn, optimizer, n_epochs: int, cuda: bool,
    train_epoch: Callable, test_epoch: Callable, filename: str, scheduler=None
):
    with open(filename, 'a') as fout:
        for epoch in range(0, n_epochs):
            train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, fout)
            fout.write(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')
            print(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')

            start = datetime.datetime.now()
            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            val_loss /= len(val_loader)
            fout.write(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f} in {datetime.datetime.now() - start}\n')
            print(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f} in {datetime.datetime.now() - start}')

            if scheduler:
                scheduler.step()


def train_triplet(train_loader: DataLoader, test_loader: DataLoader, is_double: bool = False):
    model = TripletNet() if not is_double else TripletNet().double()
    loss_fn = TripletMarginWithDistanceLoss(distance_function=PairwiseDistance(), margin=MARGIN)
    cuda = torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    loss_fn.to(device)

    fit(
        train_loader, test_loader, model, loss_fn, optimizer, NUM_EPOCHS, cuda, triplet_train_epoch,
        triplet_test_epoch, 'triplet.txt', scheduler
    )
    torch.save(model.state_dict(), '0000.pth')


def train_siamese(train_loader: DataLoader, test_loader: DataLoader, is_double: bool = False):
    model = SiameseNet() if not is_double else SiameseNet().double()
    loss_fn = ContrastiveLoss(margin=MARGIN)
    cuda = torch.cuda.is_available()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    loss_fn.to(device)

    fit(
        train_loader, test_loader, model, loss_fn, optimizer, NUM_EPOCHS, cuda, siamese_train_epoch, siamese_test_epoch,
        'siamese.txt', scheduler
    )
    torch.save(model.state_dict(), '0000.pth')


def train_classification(train_loader: DataLoader, test_loader: DataLoader, classes: List[str], is_double: bool = False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # defining the model
    model = RoadSignNet(classes=len(classes)) if not is_double else RoadSignNet(classes=len(classes)).double()
    model.to(device)

    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    # defining the loss function
    criterion = CrossEntropyLoss()

    with open('classification.txt', 'w') as fout:
        class_train(model, NUM_EPOCHS, optimizer, criterion, train_loader, device, fout)
        torch.save(model.state_dict(), '0000.pth')
        class_test(model, classes, test_loader, device, fout)
