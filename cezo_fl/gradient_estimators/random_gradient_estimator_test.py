import pytest
import torch
from torch import nn

from cezo_fl.gradient_estimators import random_gradient_estimator


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        return self.linear2(x)


@pytest.mark.parametrize(
    "rge_method",
    [
        random_gradient_estimator.RandomGradEstimateMethod.rge_forward,
        random_gradient_estimator.RandomGradEstimateMethod.rge_central,
    ],
)
@pytest.mark.parametrize("num_pert", [2, 4, 5])
def test_parameter_wise_equivalent_all_togther(
    rge_method: random_gradient_estimator.RandomGradEstimateMethod, num_pert: int
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
    rge1 = random_gradient_estimator.RandomGradientEstimator(
        model1.parameters(),
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=False,
    )
    with torch.no_grad():
        dir_grads1 = rge1.compute_grad(
            fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=54321
        )

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model2 = LinearModel()
    rge2 = random_gradient_estimator.RandomGradientEstimator(
        model2.parameters(),
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=True,
    )
    with torch.no_grad():
        dir_grads2 = rge2.compute_grad(
            fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=54321
        )

    torch.testing.assert_close(dir_grads1, dir_grads2)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)


def test_update_model_given_seed_and_grad():
    # Make the update second times and the output suppose to be the same.
    ouputs = []
    for _ in range(2):
        torch.manual_seed(0)
        fake_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

        optim = torch.optim.SGD(fake_model.parameters(), lr=1e-3)
        rge = random_gradient_estimator.RandomGradientEstimator(
            fake_model.parameters(),
            num_pert=2,
            paramwise_perturb=False,
        )
        rge.update_model_given_seed_and_grad(
            optim,
            iteration_seeds=[1, 2, 3],
            iteration_grad_scalar=[  # two perturbations
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3, 0.4]),
                torch.tensor([0.5, 0.6]),
            ],
        )
        ouputs.append(
            fake_model(torch.tensor([list(range(i, 10 + i)) for i in range(3)], dtype=torch.float))
        )
    assert ouputs[0].shape == (3, 2)
    assert ouputs[1].shape == (3, 2)
    torch.testing.assert_close(ouputs[0], ouputs[1])
