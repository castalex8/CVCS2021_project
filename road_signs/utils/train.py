import random
import os
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
from torch.autograd import Variable

TRAINING_SIZE = 3000


def load_split(basePath, csvPath):
    # initialize the list of data and labels
    data = []
    labels = []

    # load the contents of the CSV file, remove the first line (since it contains the CSV header),
    # and shuffle the rows (otherwise all examples of a particular class will be in sequential order)
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    # loop over the rows of the CSV file
    for (index, row) in enumerate(rows[:TRAINING_SIZE]):

        # check to see if we should show a status update
        if index > 0 and index % 1000 == 0:
            print(f"[INFO] processed {index} total images")

        # split the row into components and then grab the class ID and image path
        (label, imagePath) = row.strip().split(",")[-2:]

        # derive the full path to the image file and load it
        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)

        # resize the image to be 32x32 pixels, ignoring aspect ratio, and then perform
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        # update the list of data and labels, respectively
        data.append(image)
        labels.append(int(label))

    # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # return a tuple of the data and labels
    return data, labels


def trainNN(model, optimizer, criterion, train_x, train_y, val_x, val_y):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    # if torch.cuda.is_available():
    #     x_train = x_train.cuda()
    #     y_train = y_train.cuda()
    #     x_val = x_val.cuda()
    #     y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    # train_losses.append(loss_train.item())
    # val_losses.append(loss_val.item())

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), loss_val.item()
