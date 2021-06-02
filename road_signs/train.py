# set the matplotlib backend so figures can be saved in the background
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from road_signs.cnn.RoadSignNet import RoadSignNet
from road_signs.utils.train import load_split, trainNN


# initialize the number of epochs to train for, base learning rate, and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
# BS = 64
DATASET = 'gtsrb-german-traffic-sign'


if __name__ == '__main__':
    # load the label names
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    # derive the path to the training and testing CSV files
    trainPath = os.path.sep.join([DATASET, "Train.csv"])
    testPath = os.path.sep.join([DATASET, "Test.csv"])

    # load the training and testing data
    print("[INFO] loading training and testing data...")
    (trainX, trainY) = load_split(DATASET, trainPath)
    (testX, testY) = load_split(DATASET, testPath)

    # scale data to the range of [0, 1]
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    numLabels = len(np.unique(trainY))
    train_x = torch.from_numpy(trainX).reshape(trainX.shape[0], trainX.shape[-1], trainX.shape[1], trainX.shape[2])
    train_y = torch.from_numpy(trainY)
    val_x = torch.from_numpy(testX).reshape(testX.shape[0], testX.shape[-1], testX.shape[1], testX.shape[2])
    val_y = torch.from_numpy(testY)

    # defining the model
    model = RoadSignNet(depth=train_x.shape[1], classes=numLabels)

    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=INIT_LR)

    # defining the loss function
    criterion = CrossEntropyLoss()

    # checking if GPU is available
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     criterion = criterion.cuda()

    # empty list to store training losses and validation losses
    val_losses, train_losses = [], []

    # training the model
    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val = trainNN(model, optimizer, criterion, train_x, train_y, val_x, val_y)
        val_losses.append(loss_val)
        train_losses.append(loss_train)

        if epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)

    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
