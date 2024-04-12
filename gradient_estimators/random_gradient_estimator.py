import torch
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator
from shared.metrics import TensorMetric

from enum import Enum


class GradEstimateMethod(Enum):
    forward = "forward"
    central = "central"


class RandomGradientEstimator:

    def __init__(
        self,
        params: Iterator[Parameter],
        mu=1e-3,
        n_perturbation=1,
        grad_estimate_method: GradEstimateMethod = GradEstimateMethod.central.value,
    ):
        self.params_list: list[Parameter] = list(params)
        self.params_shape: list[Tensor] = [p.shape for p in self.params_list]
        self.mu = mu
        self.n_perturbation = n_perturbation

        self.grad_estimate_method = grad_estimate_method
        self.method_func_dict = {
            GradEstimateMethod.forward.value: self._forward_method,
            GradEstimateMethod.central.value: self._central_method,
        }

    def _params_list_set_(self, to_set: list[torch.Tensor]):
        if (not isinstance(to_set, list)) or (not len(to_set) == len(self.params_list)):
            raise Exception("Current to_set does not match controlled parameters")

        for p, set_value in zip(self.params_list, to_set):
            p.set_(set_value)

    def _generate_perturbation(self):
        return [torch.randn(shape) for shape in self.params_shape]

    def _forward_method(self, batch_inputs, labels, model, criterion):
        start_params = [p.clone() for p in self.params_list]
        start_pred = model(batch_inputs)
        start_loss = criterion(start_pred, labels)

        running_grads = [
            TensorMetric(f"running_grad_{i}") for i in range(len(self.params_list))
        ]

        for i in range(self.n_perturbation):
            perturbation = self._generate_perturbation()
            perturbed_params = [
                p + self.mu * perturb
                for (p, perturb) in zip(start_params, perturbation)
            ]
            self._params_list_set_(perturbed_params)
            perturbed_pred = model(batch_inputs)
            perturbed_loss = criterion(perturbed_pred, labels)
            perturbation_direction_grad = (perturbed_loss - start_loss) / self.mu

            for param_tensor_idx, running_grad in enumerate(running_grads):
                running_grad.update(
                    perturbation_direction_grad * perturbation[param_tensor_idx]
                )

        grad = [running_grad.avg for running_grad in running_grads]
        return start_params, grad

    def _central_method(self, batch_inputs, labels, model, criterion):
        start_params = [p.clone() for p in self.params_list]
        running_grads = [
            TensorMetric(f"running_grad_{i}") for i in range(len(self.params_list))
        ]

        for i in range(self.n_perturbation):
            perturbation = self._generate_perturbation()
            # forward perturbation
            forward_perturbed_params = [
                p + self.mu * perturb
                for (p, perturb) in zip(start_params, perturbation)
            ]
            self._params_list_set_(forward_perturbed_params)
            forward_perturbed_pred = model(batch_inputs)
            forward_perturbed_loss = criterion(forward_perturbed_pred, labels)

            # backward perturbation
            backward_perturbed_params = [
                p - self.mu * perturb
                for (p, perturb) in zip(start_params, perturbation)
            ]
            self._params_list_set_(backward_perturbed_params)
            backward_perturbed_pred = model(batch_inputs)
            backward_perturbed_loss = criterion(backward_perturbed_pred, labels)

            # update perturbation gradient
            perturbation_direction_grad = (
                forward_perturbed_loss - backward_perturbed_loss
            ) / (2 * self.mu)

            for param_tensor_idx, running_grad in enumerate(running_grads):
                running_grad.update(
                    perturbation_direction_grad * perturbation[param_tensor_idx]
                )

        grad = [running_grad.avg for running_grad in running_grads]
        return start_params, grad

    def estimate_and_assign_grad(self, batch_inputs, labels, model, criterion):
        estimation_method = self.method_func_dict[self.grad_estimate_method]

        start_params, grad = estimation_method(batch_inputs, labels, model, criterion)

        # reset parameters to initial state
        self._params_list_set_(start_params)

        # assign grad to parameters
        for i, p in enumerate(self.params_list):
            p.grad = grad[i]
