import pytest
import torch
from torch import nn
from torch.optim import SGD

import gradient_estimators.random_gradient_estimator as RGE


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        return self.linear2(x)


def test_parameter_wise_equivalent_all_togther():
    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model = LinearModel()
    fake_input = torch.randn(5, 3)
    fake_label = torch.randn(5, 1)
    criterion = nn.MSELoss()
    rge1 = RGE.RandomGradientEstimator(
        model,
        num_pert=2,
        grad_estimate_method="forward",
        paramwise_perturb=False,
    )
    with torch.no_grad():
        dir_grads = rge1.compute_grad(fake_input, fake_label, criterion, seed=54321)

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model = LinearModel()
    rge2 = RGE.RandomGradientEstimator(
        model,
        num_pert=2,
        grad_estimate_method="forward",
        paramwise_perturb=True,
    )
    with torch.no_grad():
        dir_grads = rge2.compute_grad(fake_input, fake_label, criterion, seed=54321)
