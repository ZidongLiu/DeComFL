from typing import Any, Callable, Iterator, Literal, Sequence, TypeAlias

import torch
from peft import PeftModel
from torch.nn import Parameter
from transformers.models.opt.modeling_opt import OPTForCausalLM

from cezo_fl.util.language_utils import LLMBatchInput

GradEstimateMethod: TypeAlias = Literal["forward", "central"]


# TODO: split this class into abstract class and several subcalsses.
class RandomGradientEstimator:
    def __init__(
        self,
        model: torch.nn.Module | OPTForCausalLM | PeftModel,
        parameters: Iterator[Parameter] | None = None,
        mu=1e-3,
        num_pert=1,
        grad_estimate_method: GradEstimateMethod = "central",
        normalize_perturbation: bool = False,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float32,
        prune_mask_arr: torch.Tensor | None = None,
        paramwise_perturb: bool = False,
        sgd_only_no_optim: bool = False,
        generation_mode: bool = False,
        generation_mode_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        if parameters is None:
            parameters = model.parameters()
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(f"trainable model size: {self.total_dimensions}")

        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype
        self.grad_estimate_method: GradEstimateMethod = grad_estimate_method

        self.paramwise_perturb = paramwise_perturb
        if paramwise_perturb:
            assert prune_mask_arr is None
            assert normalize_perturbation is False

        self.sgd_only_no_optim = sgd_only_no_optim
        if sgd_only_no_optim:
            assert self.paramwise_perturb

        self.normalize_perturbation = normalize_perturbation
        self.prune_mask_arr = None
        if prune_mask_arr:
            self.set_prune_mask(prune_mask_arr)

        # For the forward function used in generation mode usage
        self.generation_mode = generation_mode
        self.generation_mode_kwargs = generation_mode_kwargs if generation_mode_kwargs else {}

    # TODO(zidong) move this func out of this class
    def model_forward(
        self,
        batch_inputs: torch.Tensor | LLMBatchInput,
    ):
        if self.generation_mode:
            if not isinstance(self.model, (OPTForCausalLM, PeftModel)):
                raise ValueError(
                    "The model for generation_mode must be OPTForCausalLM or peft model"
                )
            generation_mode_kwargs = self.generation_mode_kwargs
            # We may need a copy for dynamic modification
            if "max_new_tokens" in self.generation_mode_kwargs:
                generation_mode_kwargs = self.generation_mode_kwargs.copy()
                assert "max_length" in generation_mode_kwargs  # both should be specified.
                # Dynamic adjust the max_new_tokens according to input length
                generation_mode_kwargs["max_new_tokens"] = min(
                    generation_mode_kwargs["max_new_tokens"],
                    generation_mode_kwargs["max_length"] - batch_inputs.input_ids.size(1),
                )
                del generation_mode_kwargs["max_length"]
            return self.model.generate(
                batch_inputs.input_ids,  # attention_mask is not needed for generation model.
                **generation_mode_kwargs,
            )
        if self.generation_mode:
            logits, loss = self.model(input_ids=batch_inputs.input_ids, labels=batch_inputs.label)
            return loss
        if isinstance(self.model, (OPTForCausalLM, PeftModel)):
            return self.model(
                input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask
            )
        elif isinstance(self.model, torch.nn.Module):
            return self.model(batch_inputs)
        else:
            raise Exception("This model type is not supported")

    def set_prune_mask(self, prune_mask_arr) -> None:
        self.prune_mask_arr = prune_mask_arr

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        p = torch.randn(
            self.total_dimensions, device=self.device, dtype=self.torch_dtype, generator=rng
        )
        if self.prune_mask_arr is not None:
            p.mul_(self.prune_mask_arr)

        if self.normalize_perturbation:
            p.div_(torch.norm(p))

        return p

    # TODO(zidong) this function should not have perturb=None usage.
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

    def generate_then_put_grad(self, seed: int, dir_grads: torch.Tensor) -> None:
        update_grad = 0
        num_pert = len(dir_grads)
        for i, dir_grad in enumerate(dir_grads):
            rng = self.get_rng(seed, i)
            update_grad += self.generate_perturbation_norm(rng).mul_(dir_grad / num_pert)
        self.put_grad(update_grad)

    def compute_grad(self, batch_inputs, labels, criterion, seed: int) -> torch.Tensor:
        if not self.paramwise_perturb:
            # We generate the perturbation vector all together. It should be faster but consume
            # more memory
            grad, perturbation_dir_grads = self._zo_grad_estimate(
                batch_inputs, labels, criterion, seed
            )
            self.put_grad(grad)
        else:
            perturbation_dir_grads = self._zo_grad_estimate_paramwise(
                batch_inputs, labels, criterion, seed
            )
            self.generate_then_put_grad_paramwise(seed, perturbation_dir_grads)

        return perturbation_dir_grads

    def sgd_no_optim_update_model(
        self, perturbation_dir_grads: torch.Tensor, seed: int, lr: float
    ) -> None:
        num_pert = len(perturbation_dir_grads)
        for i, dir_grad in enumerate(perturbation_dir_grads):
            rng = self.get_rng(seed, i)
            for param in self.parameters_list:
                _perturb = torch.randn(
                    *param.shape, device=self.device, dtype=self.torch_dtype, generator=rng
                )
                param.data.add_(_perturb, alpha=-lr * float(dir_grad) / num_pert)

    def _zo_grad_estimate(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the zeroth-order gradient estimate.

        Return a tuple, the first element is full grad and the second is the gradient scalar.

           g_full = avg_{p} (g_p*z_p),
           where  g_p = [loss(x+mu*z_p) - loss(x)] / mu ------------------- forward approach
                  g_p = [loss(x+mu*z_p) - loss(x-mu*z_p)] / (2*mu) -------- central approach

        i.e., returning (g_full, [g_1, g_2, ..., g_p]).
        """
        grad: torch.Tensor | None = None
        dir_grads = []
        denominator_factor = 2 if self.grad_estimate_method == "central" else 1
        if self.grad_estimate_method == "forward":
            pert_minus_loss = criterion(self.model_forward(batch_inputs), labels)

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = criterion(self.model_forward(batch_inputs), labels)
            if self.grad_estimate_method == "central":
                self.perturb_model(pb_norm, alpha=-2 * self.mu)
                pert_minus_loss = criterion(self.model_forward(batch_inputs), labels)
                self.perturb_model(pb_norm, alpha=self.mu)  # Restore model
            elif self.grad_estimate_method == "forward":
                self.perturb_model(pb_norm, alpha=-self.mu)  # Restore model

            dir_grad = (pert_plus_loss - pert_minus_loss) / (self.mu * denominator_factor)
            dir_grads += [dir_grad]
            if grad is None:
                grad = pb_norm.mul_(dir_grad)
            else:
                grad.add_(pb_norm, alpha=dir_grad)

            del pb_norm

        return grad.div_(self.num_pert), torch.tensor(dir_grads, device=self.device)

    def generate_then_put_grad_paramwise(self, seed: int, dir_grads: torch.Tensor) -> None:
        num_pert = len(dir_grads)
        for i, dir_grad in enumerate(dir_grads):
            rng = self.get_rng(seed, i)
            for param in self.parameters_list:
                _perturb = torch.randn(
                    *param.shape, device=self.device, dtype=self.torch_dtype, generator=rng
                )
                if i == 0:
                    param.grad = _perturb.mul_(dir_grad / num_pert)
                else:
                    param.grad += _perturb.mul_(dir_grad / num_pert)
                del _perturb

    def perturb_model_paramwise(self, rng: torch.Generator, alpha: float | int) -> None:
        for param in self.parameters_list:
            _perturb = torch.randn(
                *param.shape, device=self.device, dtype=self.torch_dtype, generator=rng
            )
            param.add_(_perturb, alpha=alpha)
            del _perturb

    def _zo_grad_estimate_paramwise(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        dir_grads = []
        denominator_factor = 2 if self.grad_estimate_method == "central" else 1
        if self.grad_estimate_method == "forward":
            pert_minus_loss = criterion(self.model_forward(batch_inputs), labels)

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            self.perturb_model_paramwise(rng, alpha=self.mu)
            pert_plus_loss = criterion(self.model_forward(batch_inputs), labels)
            if self.grad_estimate_method == "central":
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=-2 * self.mu)
                pert_minus_loss = criterion(self.model_forward(batch_inputs), labels)
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=self.mu)  # Restore model
            elif self.grad_estimate_method == "forward":
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=-self.mu)  # Restore model
            dir_grad = (pert_plus_loss - pert_minus_loss) / (self.mu * denominator_factor)
            dir_grads += [dir_grad]
        return torch.tensor(dir_grads, device=self.device)

    def update_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        assert len(iteration_seeds) == len(iteration_grad_scalar)

        if self.sgd_only_no_optim:
            lr = optimizer.defaults["lr"]  # Assume only one parameter group with lr.
            assert self.paramwise_perturb
            for one_update_seed, one_update_grad_dirs in zip(
                iteration_seeds, iteration_grad_scalar
            ):
                self.sgd_no_optim_update_model(one_update_grad_dirs, one_update_seed, lr)
            return

        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            # We don't really need optimizer.zero_grad() here because we put grad directly.
            if self.paramwise_perturb:
                self.generate_then_put_grad_paramwise(one_update_seed, one_update_grad_dirs)
            else:
                self.generate_then_put_grad(one_update_seed, one_update_grad_dirs)
            # update model
            optimizer.step()

    def revert_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        # TODO: Support sgd_only_no_optim case.
        assert not self.sgd_only_no_optim
        assert len(iteration_seeds) == len(iteration_grad_scalar)
        try:
            assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults["momentum"] == 0
        except AssertionError:
            raise Exception("Revert only supports SGD without momentum")

        lr, weight_decay = optimizer.defaults["lr"], optimizer.defaults["weight_decay"]
        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            # We don't really need optimizer.zero_grad() here because we put grad directly.
            if self.paramwise_perturb:
                self.generate_then_put_grad_paramwise(one_update_seed, one_update_grad_dirs)
            else:
                self.generate_then_put_grad(one_update_seed, one_update_grad_dirs)

            for param in self.parameters_list:
                param.add_(param.grad, alpha=lr)  # gradient ascent instead of descent.
                if weight_decay > 0:
                    param.mul_(1 / (1 - lr * weight_decay))


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
