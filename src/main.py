import torch
from data.dataset import create_dataset, get_transforms
from data.data_split import split_dataset
from models.model import get_vgg16_model
from training.train import train_one_epoch
from training.validate import validate
from training.logs import init_log_file, log_to_csv
from config.config import DEVICE, DATA_DIR, TRAIN_SIZE, VAL_SIZE, NUM_WORKERS

def main():
    transform = get_transforms()
    dataset = create_dataset(DATA_DIR, transform)
    train, val = split_dataset(dataset, TRAIN_SIZE, VAL_SIZE)

    train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    model = get_vgg16_model(num_classes=len(dataset.classes)).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

if __name__ == "__main__":
    main()
