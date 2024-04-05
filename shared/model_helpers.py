# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:58:19 2024

@author: Zidong
"""

import torch
import torch.nn as nn
import torch.optim as optim
from os import path


class Net(nn.Module):

    def __init__(self, learning_rate=1e-3, mu=1e-5, compress=None):
        super(Net, self).__init__()

        self.grad_history = []
        self.parameter_l2_history = []

        self.compress = compress
        self.learning_rate = learning_rate
        self.mu = mu

        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def gd_train_step(self, train_input, label):
        original_pred = self.forward(train_input)
        loss1 = self.criterion(original_pred, label)
        loss1.backward()
        parameter_l2 = sum([torch.sum(param**2) for param in self.parameters()])**0.5
        self.parameter_l2_history += [parameter_l2.item()]
        updates = [-self.learning_rate * param.grad.detach().clone() for param in self.parameters()]

        self.add_model_params_(updates)
        with torch.no_grad():
            for p in self.parameters():
                p.grad.zero_()

            for p in self.parameters():
                print('grad sum', torch.sum(p.grad))

    def add_model_params_(self, add_ons: list[torch.tensor]):
        with torch.no_grad():
            for p, add_on in zip(self.parameters(), add_ons):
                p.add_(add_on)


def get_current_datetime_str():
    from datetime import datetime
    now = datetime.now()
    year, month, day, hour, minute, second = now.year, now.month, now.day, now.hour, now.minute, now.second
    return f'{year}-{month}-{day}-{hour}-{minute}-{second}'


def save_model(optimizer, model, model_prefix='simple'):
    save_path = path.join(path.dirname(path.dirname(__file__)), f'models/{model_prefix}-{get_current_datetime_str()}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def load_model(optimizer, model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()


def get_model_and_optimizer(checkpoint=None):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if checkpoint:
        load_model(optimizer, model, checkpoint)

    return model, optimizer
