import torch


def train(model, epochs, optimizer, criterion, train_loader, device):
    size = len(train_loader.dataset)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} -------------------------------")
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            img0, img1, labels = data
            if device == 'cuda':
                img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(img0, img1)

            loss = criterion(*outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f"loss: {loss:>7f}  [{i * len(data):>5d}/{size:>5d}]")
                running_loss = 0.0

    print('Finished Training')


def test(model, classes, test_loader, device):
    # prepare to count predictions for each class
    correct_val = 0
    total_val = 0

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            img0, img1, labels = data
            if device == 'cuda':
                img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()

            outputs = model(img0, img1)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    print('Accuracy of the network on the', total_val, 'val pairs in %d %%' % (100 * correct_val / total_val))

