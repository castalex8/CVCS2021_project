import torch


def train(model, epochs, optimizer, criterion, train_loader, device, fout):
    size = len(train_loader.ds)
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


def test(model, classes, test_loader, device, fout):
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
