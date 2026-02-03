#!/usr/bin/env python
# coding: utf-8
"""
tinyimagenet_cnn.py
Minimal module exposing only:
- theBestCNN class
- val_loader DataLoader
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# -------------------------------
# Device
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Load TinyImageNet Data
# -------------------------------
val_path = os.path.join('Data', 'validation-10_.pkl')

with open(val_path, 'rb') as f:
    val_data = pickle.load(f)

images_val = val_data['images']
labels_val = val_data['labels']

# Create label mapping from the validation set
all_classes = sorted(list(set(labels_val)))  # 15 classes
class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
val_labels_mapped = torch.tensor([class_to_idx[l] for l in labels_val]).long()

# Standardization statistics (computed on validation for simplicity)
images_val_float = images_val.astype(np.float32) / 255.0
mean = images_val_float.mean((0,1,2))
std  = images_val_float.std((0,1,2))

# -------------------------------
# Dataset Class
# -------------------------------
class TinyImageNetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Validation transform
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])

# Validation dataset and loader
val_dataset = TinyImageNetDataset(images_val, val_labels_mapped, val_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -------------------------------
# CNN Model
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, dropout=0.0, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2,2) if pool else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class theBestCNN(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.conv1 = ConvBlock(3, 128, dropout=0.0)
        self.conv2 = ConvBlock(128, 128, dropout=0.05, pool=False)
        self.conv3 = ConvBlock(128, 256, dropout=0.1)
        self.conv4 = ConvBlock(256, 256, dropout=0.1, pool=False)
        self.conv5 = ConvBlock(256, 512, dropout=0.15)
        self.conv6 = ConvBlock(512, 512, dropout=0.15, pool=False)
        self.conv7 = ConvBlock(512, 1024, dropout=0.2)
        self.conv8 = ConvBlock(1024, 1024, dropout=0.2, pool=False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


