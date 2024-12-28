import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.models import vgg16_bn, VGG16_BN_Weights


import os
import cv2
import time
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pydicom
import optuna

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
    
RSNA_Dataset = '~/Data/RSNA/train.csv'
DDSM_Dataset = '~/Data/DDSM/mass_case_description_train_set.csv'
DDSM2_Dataset = '~/Data/DDSM2/train+ddsm.csv'

df_train = pd.read_csv(RSNA_Dataset)
df_train.head()

df_ddsm = pd.read_csv(DDSM_Dataset)
df_ddsm2 = pd.read_csv(DDSM2_Dataset)

read_ddsm2 = pd.read_csv("~/Data/DDSM2/balanced_dataset.csv")
read_ddsm2.head()

RSNA_path = '/home/whif/Data/RSNA/train_images'
RSNA_csv = pd.read_csv("~/Data/balanced_dataset.csv")
Balanced_RSNA_DDSM = pd.read_csv("~/Data/DDSM2/balanced_dataset.csv")
RSNA_DDSM_Path = '/home/whif/Data/DDSM2/full_images'

transform = transforms.Compose(
    [
     transforms.Resize((256,256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     #transforms.RandomHorizontalFlip(0.5)
     ]
    )

class RSNA_Dataset_Visual(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_data = img_data
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        # Get the image path
        img_name = os.path.join(self.img_path, str(self.img_data.loc[index, 'patient_id']), str(self.img_data.loc[index, 'image_id']) + '.dcm')
        
      
        ds = pydicom.dcmread(img_name, force=True)
        image = ds.pixel_array  
        
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        photometric_interpretation = ds.get('PhotometricInterpretation', None)

        if photometric_interpretation == "MONOCHROME1":

        # Invert the image
            image = 255 - image

    
        image = image.astype(np.float32) / 255.0

 
        image = np.stack([image] * 3, axis=0)   
        image = torch.tensor(image, dtype=torch.float32)  

       
        if self.transform:
            image = self.transform(image)

   
        label = torch.tensor(self.img_data.loc[index, 'cancer'], dtype=torch.long)  
        return image, label
    



dataset_visual = RSNA_Dataset_Visual(RSNA_csv, RSNA_path, transform)

train, val = torch.utils.data.random_split(dataset = dataset_visual,
                                                         lengths = [2000,358], generator = torch.Generator().manual_seed(42))
 

def objective(trial):
    batch_size = trial.suggest_int('batch_size', 16, 64)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size)
    
    vgg16_bn_weights = VGG16_BN_Weights.DEFAULT
    model = vgg16_bn(weights=vgg16_bn_weights)


    for params in model.parameters():
        params.requires_grad = False
            
        # Modify the classifier to adapt to the number of classes
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([torch.nn.Linear(num_features, 1)])
        model.classifier = torch.nn.Sequential(*features)
    

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(model.parameters(), lr = lr)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 1, end_factor = 0.1, total_iters = 8)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model = model.to(device)
    
    
    n_epochs = 2
    valid_loss_min = np.inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)

    # Training loop
    # Initialize CSV files for train and validation logs
    train_log_file = "train_log5.csv"
    val_log_file = "val_log6.csv"

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
            target = target_.unsqueeze(1).float()  # Adjust for BCE
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = torch.sigmoid(outputs) > 0.5  # 0.5 threshold for binary
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
                target_t = target_t.unsqueeze(1).float()
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()

                pred_t = torch.sigmoid(outputs_t) > 0.5  # Standard threshold
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
                torch.save(model.state_dict(), 'MoCad4.pt')
                print('Network improvement detected, saving model')

        scheduler.step()
    return val_acc_epoch
        

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)

print("Best hyperparameters:", study.best_params)
print("Best trial:", study.best_trial)
print("Best accuracy:", study.best_value)