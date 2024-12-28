import os
from torchvision import datasets, transforms

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.5)
    ])

def create_dataset(data_dir, transform):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset
