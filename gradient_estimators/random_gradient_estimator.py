import torch
from torch.nn import Parameter
from typing import Callable, Iterator, TypeAlias, Literal
import transformers
from shared.language_utils import LLMBatchInput


GradEstimateMethod: TypeAlias = Literal["forward", "central"]


class RandomGradientEstimator:

    def __init__(
        self,
        model: torch.nn.Module | transformers.models.opt.modeling_opt.OPTForCausalLM,
        parameters: Iterator[Parameter] | None = None,
        mu=1e-3,
        num_pert=1,
        grad_estimate_method: GradEstimateMethod = "central",
        normalize_perturbation: bool = False,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float32,
        prune_mask_arr: torch.Tensor | None = None,
    ):
        self.model = model
        if parameters is None:
            parameters = model.parameters()
        self.parameters_list: list[Parameter] = list(parameters)
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])

        self.mu = mu
        self.num_pert = num_pert
        self.normalize_perturbation = normalize_perturbation

        self.grad_estimate_method: GradEstimateMethod = grad_estimate_method
        self.method_func_dict: dict[GradEstimateMethod, Callable] = {
            "central": self._forward_method,
            "forward": self._central_method,
        }

        self.device = device
        self.torch_dtype = torch_dtype
        self.prune_mask_arr = None
        if prune_mask_arr:
            self.set_prune_mask(prune_mask_arr)

    def model_forward(self, batch_inputs: torch.Tensor | LLMBatchInput):
        if isinstance(self.model, transformers.models.opt.modeling_opt.OPTForCausalLM):
            return self.model(
                input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask
            )
        elif isinstance(self.model, torch.nn.Module):
            return self.model(batch_inputs)
        else:
            raise Exception("This model type is not supported")

    def set_prune_mask(self, prune_mask_arr) -> None:
        self.prune_mask_arr = prune_mask_arr

    def generate_perturbation_norm(self) -> torch.Tensor:
        p = torch.randn(self.total_dimensions, device=self.device, dtype=self.torch_dtype)
        if self.prune_mask_arr is not None:
            p.mul_(self.prune_mask_arr)

        if self.normalize_perturbation:
            p.div_(torch.norm(p))

        return p

    def perturb_model(self, perturb: torch.Tensor | None = None, alpha: float | int = 1) -> None:
        start = 0
        for p in self.parameters_list:
            if perturb is not None:
                _perturb = perturb[start : (start + p.numel())]
                p.add_(_perturb.view(p.shape), alpha=alpha)
            else:
                if alpha != 1:
                    p.mul_(alpha)
            start += p.numel()

    def put_grad(self, grad: torch.Tensor) -> None:
        start = 0
        for p in self.parameters_list:
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

    def compute_grad(self, batch_inputs, labels, criterion) -> torch.Tensor:
        estimation_method = self.method_func_dict[self.grad_estimate_method]
        grad, perturbation_dir_grads = estimation_method(batch_inputs, labels, criterion)

        self.put_grad(grad)
        return perturbation_dir_grads

    def _forward_method(self, batch_inputs, labels, criterion) -> tuple[torch.Tensor, torch.Tensor]:
        grad = 0
        dir_grads = []
        initial_loss = criterion(self.model_forward(batch_inputs), labels)
        for _ in range(self.num_pert):
            pb_norm = self.generate_perturbation_norm()  # TODO add random seed

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = criterion(self.model_forward(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=-self.mu)  # Restore model

            dir_grad = (pert_plus_loss - initial_loss) / self.mu
            dir_grads += [dir_grad]
            if isinstance(grad, int):
                grad = pb_norm.mul_(dir_grad)
            else:
                grad.add_(pb_norm, alpha=dir_grad)

        return grad.div_(self.num_pert), torch.tensor(dir_grads, device=self.device)

    def _central_method(self, batch_inputs, labels, criterion) -> tuple[torch.Tensor, torch.Tensor]:
        grad = 0
        dir_grads = []
        for _ in range(self.num_pert):
            pb_norm = self.generate_perturbation_norm()  # TODO add random seed

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = criterion(self.model_forward(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=-2 * self.mu)
            pert_minus_loss = criterion(self.model_forward(batch_inputs), labels)
            self.perturb_model(pb_norm, alpha=self.mu)  # Restore model

            dir_grad = (pert_plus_loss - pert_minus_loss) / (2 * self.mu)
            dir_grads += [dir_grad]
            if isinstance(grad, int):
                grad = pb_norm.mul_(dir_grad)
            else:
                grad.add_(pb_norm, alpha=dir_grad)
        return grad.div_(self.num_pert), torch.tensor(dir_grads, device=self.device)


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
