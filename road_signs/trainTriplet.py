from torch.nn import TripletMarginWithDistanceLoss, CosineSimilarity
from torch.optim import Adam
from torch.utils.data import DataLoader
from road_signs.cnn.TripletNet import TripletNet
from road_signs.datasets.GermanTrafficSignDatasetAbs import get_classes
from road_signs.datasets.GermanTrafficSignDatasetTriplet import GermanTrafficSignDatasetTriplet
from road_signs.train.trainClassificationTriple import *


# initialize the number of epochs to train for, base learning rate, and batch size
NUM_EPOCHS = 20
INIT_LR = 1e-3
BS = 128
LOCAL = False


if __name__ == '__main__':
    # load the label names
    classes = get_classes(is_local=LOCAL)
    train_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=True, is_local=LOCAL), batch_size=BS, shuffle=True)
    test_loader = DataLoader(GermanTrafficSignDatasetTriplet(train=False, is_local=LOCAL), batch_size=BS, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # defining the model
    model = TripletNet().double()
    optimizer = Adam(model.parameters(), lr=INIT_LR)
    criterion = TripletMarginWithDistanceLoss(distance_function=CosineSimilarity())

    print(device)
    model.to(device)
    criterion.to(device)

    train(model, NUM_EPOCHS, optimizer, criterion, train_loader, device)
    torch.save(model.state_dict(), 'TripletWeights/3layer256deep.pth')
    test(model, classes, test_loader, device)
