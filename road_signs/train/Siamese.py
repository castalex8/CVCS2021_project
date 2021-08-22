import datetime
import numpy as np
import torch


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, fout):
    model.train()
    losses = []
    total_loss = 0
    batch_idx = 0
    start = datetime.datetime.now()

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = tuple(d.pin_memory().cuda() for d in data)
            target = target.cuda()

        # Compute prediction error
        outputs = model(*data)
        loss = loss_fn(*outputs, target)
        losses.append(loss.item())
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            m = f'[{datetime.datetime.now() - start}] Train: [{batch_idx * len(data[0])}/{len(train_loader.ds)}'\
                f' ({100. * batch_idx / len(train_loader):.2f}%)]\tLoss: {np.mean(losses):.6f}\n'
            print(m, end='')
            fout.write(m)
            fout.flush()
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data = tuple(d.pin_memory().cuda() for d in data)
                target = target.cuda()

            outputs = model(*data)
            loss = loss_fn(*outputs, target)
            val_loss += loss.item()

    return val_loss
