from typing import Iterator

import torch
from torch.nn import Parameter


def get_parameter_indices_for_ith_elem(i, cumsum_dimension):
    """
    When model parameters are flattened into 1d array.
    Give ith element, try to find out which parameter it belongs to.
    And also the location inside the parameter.
    """
    if i < 0 or i >= cumsum_dimension[-1]:
        raise Exception(f"Index {i} out of range from cumsum {cumsum_dimension}")
    which_parameter = torch.sum(i >= cumsum_dimension).item()
    location = i if which_parameter == 0 else i - cumsum_dimension[which_parameter - 1]
    return which_parameter, location


class CoordinateGradientEstimator:
    def __init__(
        self,
        model,
        parameters: Iterator[Parameter] | None = None,
        mu=1e-3,
        device: str | None = None,
    ):
        self.device = device
        self.model = model
        if parameters is None:
            parameters = model.parameters()
        self.parameters_list: list[Parameter] = list(parameters)

        self.flatten_parameters_list = [p.flatten() for p in self.parameters_list]
        self.parameter_dimension = torch.tensor(
            [p.numel() for p in self.parameters_list], device=self.device
        )
        self.cumsum_dimension = torch.cumsum(self.parameter_dimension, dim=0)
        self.total_dimensions = int(torch.sum(self.parameter_dimension).item())

        self.mu = mu

    def estimate_ith_parameter_grad(self, i, loss_fn, base_loss):
        index_of_parameter_in_parameters_list, index_within_parameter = (
            get_parameter_indices_for_ith_elem(i, self.cumsum_dimension)
        )
        flatten_parameter = self.flatten_parameters_list[index_of_parameter_in_parameters_list]
        # clone to be safe, might not need
        orig_value = flatten_parameter[index_within_parameter].clone()

        flatten_parameter[index_within_parameter] = orig_value + self.mu
        loss = loss_fn()
        grad_i = (loss - base_loss) / self.mu

        # reset parameter
        flatten_parameter[index_within_parameter] = orig_value
        return grad_i

    def put_grad(self, grad: torch.Tensor) -> None:
        start = 0
        for p in self.parameters_list:
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

    def get_estimate_indices(self):
        return range(self.total_dimensions)

    def compute_grad(self, batch_inputs, labels, criterion) -> None:
        grad = torch.zeros(self.total_dimensions, device=self.device)

        def loss_fn():
            return criterion(self.model(batch_inputs), labels)

        base_loss = loss_fn()

        estimate_indices = self.get_estimate_indices()
        for i in estimate_indices:
            grad[i] = self.estimate_ith_parameter_grad(i, loss_fn, base_loss)

        self.put_grad(grad)
