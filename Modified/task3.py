import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import time
import os
import copy
import timm
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
# from data.dataset import *
# from att import *
#SE Att
import pandas as pd
from PIL import Image

class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.transform(Image.open(row["path"])), row['label']
        )

# dataset = datasets.ImageFolder('path/to/data', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

class CustomSEBlock(nn.Module):
    def __init__(self, in_channels):
        super(CustomSEBlock, self).__init__()
        self.se_layer = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = self.se_layer(scale)
        scale = self.sigmoid(scale)
        return x * scale
    
class ModifiedBlock(nn.Module):
    def __init__(self, original_blocks, att_block):
        super(ModifiedBlock, self).__init__()
        # The first three layers of the original block
        self.first_blocks = nn.Sequential(*list(original_blocks)[:1]) #: 2 for timm-cnextL, 5 for tv-cnextL
        # Custom SE block
        self.att_block = att_block
        # The remaining layers of the original block
        self.last_blocks = nn.Sequential(*list(original_blocks)[1:])

    def forward(self, x):
        x = self.first_blocks(x)
        x = self.att_block(x)
        x = self.last_blocks(x)
        return x


def testing(model, test_loader_cub,device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader_cub, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# train_dataset = "data/train_t2_256.pkl" #CHANGE "data/train_t2_384.pkl"
# val_dataset = "data/test_t2_256.pkl" #CHANGE "data/test_t2_384.pkl"

# file = open(train_dataset,'rb') 
# train_loader = pickle.load(file)
# file.close()

# file = open(val_dataset,'rb') 
# test_loader = pickle.load(file)
# file.close()

data_dir = "/apps/local/shared/CV703/datasets/FoodX/food_dataset"

split = 'train'
train_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

split = 'val'
val_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# write data transform here as per the requirement
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),                     # Randomly crop and resize to 224x224
    transforms.Resize((256, 256)),                          # Resize to 224x224 (384, 384)
    # transforms.RandomHorizontalFlip(),                      # Randomly flip the image horizontally
    # transforms.RandomRotation(45),                          # Randomly rotate the image by up to 30 degrees
    # transforms.ColorJitter(),                               # Randomly change brightness, contrast, saturation, and hue
    # transforms.RandomAffine(
    # degrees = 45,  # No rotation
    # translate=(0.2, 0.2),  # Translate up to 10% horizontally and vertically
    # shear=(0, 30),  # Shear by 0 to 30 degrees
    # ),
    transforms.ToTensor(),                                  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=mean,std=std)                 # Normalize the image, Mean and std for ImageNet
])
test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

train_dataset = FOODDataset(train_df,train_transform)
val_dataset = FOODDataset(val_df, test_transform)

# load in into the torch dataloader to get variable batch size, shuffle 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

lr = 1e-5 #CHANGE
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
path = os.path.join('/home/thao.nguyen/CV703/Assignment_01/New/Modified/', f"ConvNextV2L_t3_Att12_Drop_256.pth") #CHANGE

# Move model and data to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 251
new_dropout_rate = 0.1  # Set your desired dropout rate

from datetime import datetime
import logging

log_dir = '/home/thao.nguyen/CV703/Assignment_01/New/Modified/'  # Change to your desired log directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_Att12_Drop_256_task3.log'
logging.basicConfig(filename=os.path.join(log_dir, log_filename), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# #convnextv2_large + SE
model = timm.create_model('convnextv2_large', pretrained=True) 
model.head.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(model.head.fc.in_features, num_classes))
original_blocks = model.stages  # model.stages[2] has 27 blocks
in_channels = 192
model.stages = ModifiedBlock(original_blocks, CustomSEBlock(in_channels))

for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = new_dropout_rate

# checkpoint = torch.load('/home/thao.nguyen/CV703/Assignment_01/Task2_dropout/ConvNextV2L+SE_256.pth') #CHANGE
# model = checkpoint['model']
# print(checkpoint['lr'])
# print(checkpoint['epoch'])

# Reconstruct the fully connected layer with the new dropout rate
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = lr,betas=(0.9, 0.99))
Cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
StepLR = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Train the model with tqdm for progress bar
num_epochs = 20
best_accuracy=testing(model, test_loader)
logging.info(f"Initial Best Val Accuracy: {best_accuracy}%")
print(f"Best Val Accuracy: {best_accuracy}%")
# print(f"Best Train Accuracy: {testing(model, train_loader)}%")

for epoch in range(num_epochs):
    correct = 0
    total = 0
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
    training_acc= 100 * correct / total
    if epoch < 50:
        StepLR.step()
    elif epoch > 80:
        Cosine.step()


    # print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {100 * correct / total}%")
    accuracy = testing(model, test_loader)
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {training_acc}%, Validation Accuracy: {accuracy}%")
    # Evaluate the model
    if accuracy > best_accuracy:
        best_accuracy=accuracy
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model': model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    }, path)
        print(f"Current lr: {optimizer.param_groups[0]['lr']}")
        print(f"Training Accuracy: {training_acc}%")
        print(f"Best Val Accuracy: {best_accuracy}%")
    # print(f"Test Accuracy: {accuracy}%")