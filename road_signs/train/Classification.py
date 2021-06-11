import torch


def train(model, epochs, optimizer, criterion, train_loader, device):
    size = len(train_loader.dataset)
    for epoch in range(epochs):
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
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f"loss: {loss:>7f}  [{i * len(inputs):>5d}/{size:>5d}]")
                running_loss = 0.0

    print('Finished Training')


def test(model, classes, test_loader, device):
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
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
