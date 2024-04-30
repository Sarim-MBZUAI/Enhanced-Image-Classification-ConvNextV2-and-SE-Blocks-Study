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



class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,*args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content

class FGVCAircraft(VisionDataset):
    """
    FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None):
        super(FGVCAircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]
        
        # Modify class index as we are going to concat to CUB dataset
        num_cub_classes = 200 #num_cub_classes = len(train_dataset_cub.class_to_idx)
        targets = [t + num_cub_classes for t in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images    

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
        self.first_blocks = nn.Sequential(*list(original_blocks)[:2]) #: 2 for timm-cnextL, 5 for tv-cnextL
        # Custom SE block
        self.att_block = att_block
        # The remaining layers of the original block
        self.last_blocks = nn.Sequential(*list(original_blocks)[2:])

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


data_root = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011"
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
train_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=train_transform, split="train")
test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=train_transform, split="test")
# load in into the torch dataloader to get variable batch size, shuffle 
train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=16, shuffle=True, num_workers=4, pin_memory=True) #24
test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=16, shuffle=False, num_workers=4, pin_memory=True) #24

data_root = "/apps/local/shared/CV703/datasets/fgvc-aircraft-2013b"
train_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=train_transform, train=True)
test_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=train_transform, train=False)
# load in into the torch dataloader to get variable batch size, shuffle 
train_loader_aircraft = torch.utils.data.DataLoader(train_dataset_aircraft, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader_aircraft = torch.utils.data.DataLoader(test_dataset_aircraft, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

from torch.utils.data import ConcatDataset 
concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

train_loader = torch.utils.data.DataLoader(
             concat_dataset_train,
             batch_size=16, shuffle=True, num_workers=4, pin_memory=True
            )
test_loader = torch.utils.data.DataLoader(
             concat_dataset_test,
             batch_size=16, shuffle=False, num_workers=4, pin_memory=True
            )

lr = 1e-5 #CHANGE
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
path = os.path.join('/home/thao.nguyen/CV703/Assignment_01/L6020', f"ConvNextV2L_t2_256.pth") #CHANGE

# Move model and data to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 300
new_dropout_rate = 0.1  # Set your desired dropout rate

from datetime import datetime
import logging

log_dir = '/home/thao.nguyen/CV703/Assignment_01/L6020'  # Change to your desired log directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_256_task2.log'
logging.basicConfig(filename=os.path.join(log_dir, log_filename), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# #convnextv2_large + SE
model = timm.create_model('convnextv2_large', pretrained=True) 
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

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
num_epochs = 100
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