# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:08:23 2024

@author: Zidong
"""

from shared.simple_split_model import SplitSimpleCNN
# from shared.model_helpers import Net

import torch
import torchvision
# Define transforms
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

model = SplitSimpleCNN()

# count = 0
# with torch.no_grad():
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         model.gd_train_step(inputs, labels)
#         loss = model.criterion(model(inputs), labels)
#         if torch.isnan(loss).all():
#             break
#         if i % 100 == 0:

#             running_loss += loss.item()
#             print('[%5d] loss: %.3f' % (i + 1, running_loss / 100))
#             running_loss = 0.0

# for p in model.parameters():
#     p.grad.zero_()

# original_pred = model.forward(inputs)
# loss1 = model.criterion(original_pred, labels)
# loss1.backward()
# with torch.no_grad():
#     for p in model.parameters():
#         p -= p.grad * lr
#     model.zero_grad()

if __name__ == '__main__':
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        loss = model.sgd_train_step(inputs, labels)
        # if torch.isnan(loss).all():
        #     break
        running_loss += loss
        if i % 100 == 99:
            print('[%5d] loss: %.3f' % (i + 1, running_loss / 100))
            running_loss = 0.0

    plt.plot(model.parameter_l2_history)
