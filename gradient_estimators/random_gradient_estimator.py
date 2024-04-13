import torch
from torch.nn import Parameter
from typing import Iterator, Optional

from enum import Enum


class GradEstimateMethod(Enum):
    forward = "forward"
    central = "central"


class RandomGradientEstimator:

    def __init__(
        self,
        model,
        parameters: Optional[Iterator[Parameter]] = None,
        mu=1e-3,
        num_pert=1,
        grad_estimate_method: GradEstimateMethod = GradEstimateMethod.central.value,
        normalize_perturbation: bool = False,
        device: Optional[str] = None,
    ):
        self.model = model
        if parameters is None:
            parameters = model.parameters()
        self.parameters_list: list[Parameter] = list(parameters)
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])

        self.mu = mu
        self.num_pert = num_pert
        self.normalize_perturbation = normalize_perturbation

        self.grad_estimate_method = grad_estimate_method
        self.method_func_dict = {
            GradEstimateMethod.forward.value: self._forward_method,
            GradEstimateMethod.central.value: self._central_method,
        }

        self.device = device

    def generate_perturbation_norm(self) -> torch.Tensor:
        p = torch.randn(self.total_dimensions)

        if self.device is not None:
            p = p.to(self.device)

        if self.normalize_perturbation:
            return p / torch.norm(p)
        else:
            return p

    def perturb_model(self, perturb: torch.Tensor, *, alpha: float | int = 1) -> None:
        start = 0
        for p in self.parameters_list:
            _perturb = perturb[start : (start + p.numel())]
            p.add_(_perturb.view(p.shape), alpha=alpha)
            start += p.numel()

    def put_grad(self, grad: torch.Tensor) -> None:
        start = 0
        for p in self.parameters_list:
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

    def compute_grad(self, batch_inputs, labels, criterion) -> None:
        estimation_method = self.method_func_dict[self.grad_estimate_method]
        grad = estimation_method(batch_inputs, labels, criterion)

        self.put_grad(grad)

    def _forward_method(self, batch_inputs, labels, criterion):
        grad = 0
        initial_loss = criterion(self.model(batch_inputs), labels)
        for _ in range(self.num_pert):
            pb_norm = self.generate_perturbation_norm()  # TODO add random seed

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = criterion(self.model(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=-self.mu)  # Restore model

            dir_grad = (pert_plus_loss - initial_loss) / self.mu
            grad += dir_grad * pb_norm
        return grad / self.num_pert

    def _central_method(self, batch_inputs, labels, criterion):
        grad = 0
        for _ in range(self.num_pert):
            pb_norm = self.generate_perturbation_norm()  # TODO add random seed

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = criterion(self.model(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=-2 * self.mu)
            pert_minus_loss = criterion(self.model(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=self.mu)  # Restore model

            dir_grad = (pert_plus_loss - pert_minus_loss) / (2 * self.mu)
            grad += dir_grad * pb_norm
        return grad / self.num_pert
