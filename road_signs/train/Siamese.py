import numpy as np
import torch


def train_epoch(train_loader, model, loss_fn, optimizer, cuda):
    model.train()
    losses = []
    total_loss = 0
    batch_idx = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = tuple(d.cuda() for d in data)
            target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        loss_inputs = outputs
        target = (target,)
        loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 99:
            print(
                f'Train: [{batch_idx * len(data[0])}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader)}%)]\tLoss: {np.mean(losses)}'
            )
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if cuda:
                data = tuple(d.cuda() for d in data)
                target = target.cuda()

            outputs = model(*data)

            loss_inputs = outputs
            target = (target,)
            loss_inputs += target
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs
            val_loss += loss.item()

    return val_loss
