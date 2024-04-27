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
        prune_mask_arr: Optional[torch.Tensor] = None,
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
        self.prune_mask_arr = None
        if prune_mask_arr:
            self.set_prune_mask(prune_mask_arr)

    def set_prune_mask(self, prune_mask_arr):
        self.prune_mask_arr = prune_mask_arr

    def generate_perturbation_norm(self) -> torch.Tensor:
        p = torch.randn(self.total_dimensions, device=self.device)
        if self.prune_mask_arr is not None:
            p.mul_(self.prune_mask_arr)

        if self.normalize_perturbation:
            p.div_(torch.norm(p))

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


# Copied from DeepZero and slightly modified
@torch.no_grad()
def functional_forward_rge(func, params_dict: dict, num_pert, mu):
    base = func(params_dict)
    grads_dict = {}
    for _ in range(num_pert):
        perturbs_dict, perturbed_params_dict = {}, {}
        for key, param in params_dict.items():
            perturb = torch.randn_like(param)
            perturb /= torch.norm(perturb) + 1e-8
            perturb *= mu
            perturbs_dict[key] = perturb
            perturbed_params_dict[key] = perturb + param
        directional_derivative = (func(perturbed_params_dict) - base) / mu
        if len(grads_dict.keys()) == len(params_dict.keys()):
            for key, perturb in perturbs_dict.items():
                grads_dict[key] += perturb * directional_derivative / num_pert
        else:
            for key, perturb in perturbs_dict.items():
                grads_dict[key] = perturb * directional_derivative / num_pert
    return grads_dict
