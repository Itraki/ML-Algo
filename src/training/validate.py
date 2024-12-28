import torch

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    total = len(val_loader.dataset)

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc
