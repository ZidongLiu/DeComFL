import torch
import torch.nn as nn


class LinearRegSlope(nn.Module):

    def __init__(self):
        super(LinearRegSlope, self).__init__()
        self.slope = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        return x * self.slope


class LinearRegIntercept(nn.Module):

    def __init__(self):
        super(LinearRegIntercept, self).__init__()
        self.intercept = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        return x + self.intercept
