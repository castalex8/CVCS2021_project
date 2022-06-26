from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader


def train(model, epochs, optimizer, criterion, train_loader, device, fout):
    size = len(train_loader)
    for epoch in range(epochs):
        fout.write(f"Epoch {epoch + 1} -------------------------------\n")
        print(f"Epoch {epoch + 1} -------------------------------")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if str(device) == 'cuda':
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                fout.write(f"loss: {loss:>7f}  [{i * len(inputs):>5d}/{size:>5d}]\n")
                print(f"loss: {loss:>7f}  [{i * len(inputs):>5d}/{size:>5d}]")
                running_loss = 0.0

    fout.write('Finished Training')
    print('Finished Training')


def test(model: nn.Module, classes: List[str], test_loader: DataLoader, device: torch.device, fout):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if str(device) == 'cuda':
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        fout.write("Accuracy for class {:5s} is: {:.1f} %\n".format(classname, accuracy))
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


def test_pedestrian(
    model: nn.Module, classes: List[str], test_loader: DataLoader, device: torch.device, pedestrian_label: str, fout
):
    total_pred = .0
    tp = .0
    fp = .0
    tn = .0
    fn = .0

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if str(device) == 'cuda':
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                label_class = classes[label]
                pred_class = classes[prediction]

                if pedestrian_label in label_class or pedestrian_label in pred_class:
                    if pedestrian_label in label_class:
                        if label_class == pred_class:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if label_class == pred_class:
                            tp += 1
                        else:
                            fn += 1
                else:
                    tn += 1

                total_pred += 1

    # print accuracy for each class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = 100 * (tp + tn) / total_pred
    fscore = (2 * precision * recall) / (precision + recall)
    fout.write(f"Accuracy: {accuracy:.2f}% Precision: {precision:.2f} Recall: {recall:.2f} Fscore: {fscore:.2f}")
    print(f"Accuracy: {accuracy:.2f}% Precision: {precision:.2f} Recall: {recall:.2f} Fscore: {fscore:.2f}")
