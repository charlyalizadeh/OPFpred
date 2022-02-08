import torch
from torch import tensor
import torch.nn as nn
from torch.optim import Adam, RMSprop

optimizers = {
    'Adam': Adam,
    'RMSprop': RMSprop
}
losses = {
    'crossentropy': nn.CrossEntropyLoss,
    'binary_crossentropy': nn.BCELoss,
    'mse': nn.MSELoss,
    'absolute': nn.L1Loss
}


def train(model, train_loader, val_loader, epochs, optimizer, criterion, lr):
    optimizer = optimizers[optimizer](model.parameters(), lr=lr)
    criterion = losses[criterion]()
    train_losses = tensor([]).cpu()
    val_losses = tensor([]).cpu()

    for epoch in range(epochs):
        train_step(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            train_loss = score(model, train_loader, criterion)
            val_loss = score(model, val_loader, criterion)
            train_losses = torch.cat((train_losses, torch.tensor([train_loss.cpu()])), 0)
            val_losses = torch.cat((val_losses, torch.tensor([val_loss.cpu()])), 0)
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


def train_step(model, loader, criterion, optimizer):
    model.train()
    for i, (X, y) in enumerate(loader):
        print(f"    {i + 1} / {len(loader)}", end="\r")
        X, y = X.to('cuda'), y.to('cuda')
        out = torch.flatten(model(X))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print()


def predict(model, loader, to_numpy=False):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for X, y in loader:
            X, y = X.to('cuda'), y.to('cuda')
            out = torch.flatten(model(X))
            y_pred = torch.cat((y_pred, out), 0)
            y_true = torch.cat((y_true, y), 0)
    if to_numpy:
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    return y_true, y_pred


def score(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        y_true, y_pred = predict(model, loader)
        return criterion(y_pred, y_true)
