import torch
from datetime import datetime

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    total = len(train_loader.dataset)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc
