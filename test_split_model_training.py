# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:08:23 2024

@author: Zidong
"""

from shared.simple_split_model import SplitSimpleCNN



import torch
from torch import nn
import torchvision
# Define transforms
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

model = SplitSimpleCNN()

count = 0
with torch.no_grad():
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        model.single_train_step(inputs, labels)
        
        if i % 100 == 0:
            loss = model.criterion(model(inputs), labels)
            running_loss += loss.item()
            print('[%5d] loss: %.3f' % (i + 1, running_loss / 100))
            running_loss = 0.0