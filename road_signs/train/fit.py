def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, train_epoch, test_epoch, filename):
    with open(filename, 'a') as fout:
        for epoch in range(0, n_epochs):
            train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, fout)
            fout.write(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')
            print(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')

            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            val_loss /= len(val_loader)
            fout.write(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}\n')
            print(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}')

            scheduler.step()


def fitNoSched(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, cuda, train_epoch, test_epoch, filename):
    with open(filename, 'a') as fout:
        for epoch in range(0, n_epochs):
            train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, fout)
            fout.write(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')
            print(f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}')

            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            val_loss /= len(val_loader)
            fout.write(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}\n')
            print(f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}')
