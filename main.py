# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:21:51 2024

@author: Zidong
"""
import numpy as np
from model_helpers import get_model_and_optimizer
import torchvision.transforms as transforms
import torchvision
from torch_attack_class import Attack
from time import time
import ssl
from os import path

ssl._create_default_https_context = ssl._create_unverified_context

model, optimizer = get_model_and_optimizer(path.join(path.dirname(__file__), 'models/simple-2024-3-24-20-45-56.pt'))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

model_attacker = Attack(model)

np.random.seed(1)
result, result_images, losses = [], [], []

for i in range(3):
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
