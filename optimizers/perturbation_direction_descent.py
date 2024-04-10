import torch
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator


class PDD:

    def __init__(
        self,
        params: Iterator[Parameter],
        lr=1e-3,
        mu=1e-3,
        grad_estimate_method="forward",
    ):
        self.params_list: list[Parameter] = list(params)
        self.params_shape: list[Tensor] = [p.shape for p in self.params_list]
        self.lr = lr
        self.mu = mu
        self.grad_estimate_method = grad_estimate_method

        self.current_perturbation = None

    def _params_list_add_(self, to_add: list[torch.Tensor]):
        if (not isinstance(to_add, list)) or (not len(to_add) == len(self.params_list)):
            raise Exception("Current to_add does not match controlled parameters")

        for p, added_value in zip(self.params_list, to_add):
            p.add_(added_value)

    def generate_perturbation(self):
        self.current_perturbation = [torch.randn(shape) for shape in self.params_shape]
        return self.current_perturbation

    def apply_perturbation_1(self):
        if self.grad_estimate_method == "forward":
            return

        if self.grad_estimate_method == "middle":
            self._params_list_add_(
                [-self.mu * perturb for perturb in self.current_perturbation]
            )
            return

        return

    def apply_perturbation_2(self):
        if self.grad_estimate_method == "forward":
            self._params_list_add_(
                [self.mu * perturb for perturb in self.current_perturbation]
            )

        if self.grad_estimate_method == "middle":
            self._params_list_add_(
                [2 * self.mu * perturb for perturb in self.current_perturbation]
            )
            return

        return

    def step(self, grad, to_cancel_perturbation=True):
        if (not isinstance(self.current_perturbation, list)) or (
            not len(self.current_perturbation) == len(self.params_list)
        ):
            raise Exception("Current perturbation does not match controlled parameters")

        # update_parameters, need to minus perturbation(since model is already changed)
        # then move to new_direction
        # x_t+1 = x_t - learning_rate * grad * perturbation
        # x_t+0.5 = x_t + mu * perturbation
        # x_t+1 = x_t+0.5 - mu * perturbation - learning_rate * grad * perturbation
        # x_t+1 = x_t+0.5 - (mu + learning_rate * grad) * perturbation
        if to_cancel_perturbation:
            perturb_multiplier = -(self.mu + self.lr * grad)
        else:
            perturb_multiplier = -self.lr * grad

        for p, perturb in zip(self.params_list, self.current_perturbation):
            p.add_(perturb_multiplier * perturb)

    @property
    def divider(self):
        if self.grad_estimate_method == "forward":
            return self.mu

        if self.grad_estimate_method == "middle":
            return self.mu * 2

    def calculate_grad(self, perturbation_1_loss, perturbation_2_loss):
        return (perturbation_2_loss - perturbation_1_loss) / self.divider
