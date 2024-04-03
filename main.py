# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:21:51 2024

@author: Zidong
"""
import numpy as np
from model_helpers import get_model_and_optimizer

model, optimizer = get_model_and_optimizer(
    r'C:\research\zoo_attack\models\simple-2024-3-24-20-45-56.pt')
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import torchvision

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=False,
                                        transform=transform)


from torch_attack_class import Attack

model_attacker = Attack(model)

np.random.seed(1)
result, result_images, losses = [], [], []

from time import time


for i in range(30):
    image, label = trainset[i]
    model_pred = model(image).argmax().item()
    if label == model_pred:
        t1 = time()
        is_success, attack_image, loss = model_attacker.attack(image, label)
        result += [is_success]
        result_images += [(image, attack_image)]
        losses += [loss]
        t2 = time()
        print(i, (t2 - t1) / (len(loss) + 1))