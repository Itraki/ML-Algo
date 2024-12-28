import os
import cv2
import time
import csv
import torch
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets
import pydicom
import optuna
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torchvision.transforms as transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)


transform = transforms.Compose(
    [
    transforms.ToTensor(),
     transforms.Resize((256,256)),
     transforms.Normalize(mean=[0.5], std=[0.5]),
     transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(0.5)
     ]
    )


NUM_WORKERS = 2

data_dir = "~/Data/RSNA/processed_images"

    # Use ImageFolder to create dataset(s)
dataset = datasets.ImageFolder(data_dir, transform=transform)

print(len(dataset))
    # Get class names
class_names = dataset.classes
num_classes = len(class_names)



train, val = torch.utils.data.random_split(dataset = dataset,
                                                         lengths = [2000,358], generator = torch.Generator().manual_seed(42))
 

def objective(trial):
    batch_size = trial.suggest_int('batch_size', 16, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    #momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    
    train_loader = DataLoader(
        train,
        batch_size=16,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    val_loader = DataLoader(
        val,
        batch_size=16,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
   
    
    vgg16_bn_weights = VGG16_BN_Weights.DEFAULT
    model = vgg16_bn(weights=vgg16_bn_weights)


    for params in model.parameters():
        params.requires_grad = True
            
        # Modify the classifier to adapt to the number of classes
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([torch.nn.Linear(num_features, num_classes)])
        model.classifier = torch.nn.Sequential(*features)
    
    #Adding dropOut
    feats_list = list(model.features)
    new_feats_list = []
    for feat in feats_list:
        new_feats_list.append(feat)
        if isinstance(feat, torch.nn.Conv2d):
            new_feats_list.append(torch.nn.Dropout(p=0.3, inplace=True))
    
    model.features = torch.nn.Sequential(*new_feats_list)
            
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 1, end_factor = 0.1, total_iters = 8)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model = model.to(device)
    
    
    n_epochs = 5
    valid_loss_min = np.inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)

    # Training loop
    # Initialize CSV files for train and validation logs
    train_log_file = "train_log16.csv"
    val_log_file = "val_log17.csv"

    # Initialize CSV headers if files are not present
    for file_name in [train_log_file, val_log_file]:
        if not os.path.exists(file_name):
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Loss", "Accuracy", "Timestamp"])

    # Training loop
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()

        print("Epoch ", epoch)
        for batch_idx, (data_, target_) in enumerate(train_loader):
            data_, target_ = data_.to(device), target_.to(device)

            outputs = model(data_)
            target = target_  # Adjust for BCE
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            accuracy = (pred == target).sum().item() / target.size(0)

            if batch_idx % 3 == 0:
                print(f'Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], Loss: {loss.item():.4f}')

        # Calculate and log training loss and accuracy for the epoch
        train_loss_epoch = running_loss / total_step
        train_acc_epoch = 100 * accuracy
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)

        # Log to CSV
        with open(train_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss_epoch, train_acc_epoch, timestamp])

        print(f'\nTrain loss: {train_loss_epoch:.4f}, Train acc: {train_acc_epoch:.4f}')

        # Validation step
        with torch.no_grad():
            model.eval()
            batch_loss = 0
            for data_t, target_t in val_loader:
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = model(data_t)
                target_t = target_t
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()

                pred_t = torch.argmax(torch.softmax(outputs_t, dim=1), dim=1)
                accuracy_t = (target_t == pred_t).sum().item() / target_t.size(0)

            # Calculate and log validation loss and accuracy for the epoch
            val_loss_epoch = batch_loss / len(val_loader)
            val_acc_epoch = 100 * accuracy_t
            val_loss.append(val_loss_epoch)
            val_acc.append(val_acc_epoch)

            # Log to CSV
            with open(val_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, val_loss_epoch, val_acc_epoch, timestamp])

            print(f'Validation loss: {val_loss_epoch:.4f}, Validation acc: {val_acc_epoch:.4f}\n')

            # Save model if validation improves
            if batch_loss < valid_loss_min:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'MoCaD8.pt')
                print('Network improvement detected, saving model')

        scheduler.step()
    return val_acc_epoch
        

study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3", study_name="training_JPEG_IMAGES_43")
study.optimize(objective, n_trials=1)

print("Best hyperparameters:", study.best_params)
print("Best trial:", study.best_trial)
print("Best accuracy:", study.best_value)