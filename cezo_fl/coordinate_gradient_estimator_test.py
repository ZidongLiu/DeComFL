import pytest
import torch
from torch import nn
from torch.optim import SGD

from cezo_fl.coordinate_gradient_estimator import CoordinateGradientEstimator as CGE
from cezo_fl.coordinate_gradient_estimator import get_parameter_indices_for_ith_elem


class LinearReg(nn.Module):
    def __init__(self):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_get_parameter_indices_for_ith_elem():
    # parameters size:   [5, 10, 4, 12]
    # parameters cumsum: [5, 15, 19, 31]
    parameter_size = torch.tensor([5, 10, 4, 12])
    parameter_cumsum = torch.cumsum(parameter_size, dim=0)

    assert get_parameter_indices_for_ith_elem(3, parameter_cumsum) == (0, 3)
    assert get_parameter_indices_for_ith_elem(4, parameter_cumsum) == (0, 4)
    assert get_parameter_indices_for_ith_elem(5, parameter_cumsum) == (1, 0)
    assert get_parameter_indices_for_ith_elem(12, parameter_cumsum) == (1, 7)
    assert get_parameter_indices_for_ith_elem(16, parameter_cumsum) == (2, 1)
    assert get_parameter_indices_for_ith_elem(30, parameter_cumsum) == (3, 11)

    with pytest.raises(Exception):
        get_parameter_indices_for_ith_elem(31, parameter_cumsum)
        get_parameter_indices_for_ith_elem(-1, parameter_cumsum)


def test_get_estimate_indices():
    model = LinearReg()
    cge = CGE(model)

    assert cge.get_estimate_indices() != range(3)
    assert cge.get_estimate_indices() == range(2)


def test_simple_model_training():
    torch.manual_seed(1)
    model = LinearReg()

    xs = torch.arange(-5, 5, 0.1).reshape(-1, 1)
    n = len(xs)
    true_slope = 2
    true_intercept = 1
    ys = xs * true_slope + true_intercept

    cge = CGE(model)
    sgd = SGD(model.parameters(), momentum=0.8)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(1000):
            start_i = i % n
            batch_x = xs[start_i : (start_i + 5)]
            batch_y = ys[start_i : (start_i + 5)]

            cge.compute_grad(batch_x, batch_y, criterion)

            sgd.step()

    model_parameters = list(model.parameters())
    estimated_slope = model_parameters[0].flatten()[0].item()
    estimated_intercept = model_parameters[1].flatten()[0].item()

    assert (
        abs(estimated_slope - true_slope) < 0.01
        and abs(estimated_intercept - true_intercept) < 0.01
    )
