def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, train_epoch, test_epoch):
    for epoch in range(0, n_epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda)
        print(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')

        val_loss = test_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)
        print(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}')

        scheduler.step()
