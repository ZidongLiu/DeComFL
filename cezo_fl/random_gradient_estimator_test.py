import pytest
import torch
from torch import nn

import cezo_fl.random_gradient_estimator as RGE


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        return self.linear2(x)


@pytest.mark.parametrize("rge_method", ["forward", "central"])
@pytest.mark.parametrize("num_pert", [2, 4, 5])
def test_parameter_wise_equivalent_all_togther(
    rge_method: RGE.GradEstimateMethod, num_pert: int
) -> None:
    """
    NOTE: Do not extend this test for large model. This test only works when model is small.
    To be specific, works number of parameters <= 10.
    """
    fake_input = torch.randn(5, 3)
    fake_label = torch.randn(5, 1)
    criterion = nn.MSELoss()

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model1 = LinearModel()
    rge1 = RGE.RandomGradientEstimator(
        model1,
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=False,
    )
    with torch.no_grad():
        dir_grads1 = rge1.compute_grad(fake_input, fake_label, criterion, seed=54321)

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model2 = LinearModel()
    rge2 = RGE.RandomGradientEstimator(
        model2,
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=True,
    )
    with torch.no_grad():
        dir_grads2 = rge2.compute_grad(fake_input, fake_label, criterion, seed=54321)

    torch.testing.assert_close(dir_grads1, dir_grads2)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)
