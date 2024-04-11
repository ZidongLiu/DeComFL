import torch
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator
from shared.metrics import TensorMetric


class RGE_SGD:

    def __init__(self, params: Iterator[Parameter], lr=1e-3, mu=1e-3, n_permutation=1):
        self.params_list: list[Parameter] = list(params)
        self.params_shape: list[Tensor] = [p.shape for p in self.params_list]
        self.lr = lr
        self.mu = mu
        self.n_permutation = n_permutation

    def _params_list_set_(self, to_set: list[torch.Tensor]):
        if (not isinstance(to_set, list)) or (not len(to_set) == len(self.params_list)):
            raise Exception("Current to_set does not match controlled parameters")

        for p, set_value in zip(self.params_list, to_set):
            p.set_(set_value)

    def generate_perturbation(self):
        return [torch.randn(shape) for shape in self.params_shape]

    def step(self, batch_inputs, labels, model, criterion):
        start_params = [p.clone() for p in self.params_list]
        start_pred = model(batch_inputs)
        start_loss = criterion(start_pred, labels)

        running_grads = [
            TensorMetric(f"running_grad_{i}") for i in range(len(self.params_list))
        ]

        for i in range(self.n_permutation):
            perturbation = self.generate_perturbation()
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

        updated_params = [
            p - self.lr * grad_p for (p, grad_p) in zip(start_params, grad)
        ]
        self._params_list_set_(updated_params)
