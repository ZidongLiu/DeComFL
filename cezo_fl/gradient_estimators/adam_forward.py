from typing import Callable, Iterator, Sequence
from cezo_fl.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator

import torch
from torch.nn import Parameter

from enum import Enum


class KUpdateStrategy(Enum):
    ALL_LOCAL_UPDATES = "all_local_updates"
    LAST_LOCAL_UPDATE = "last_local_update"


class AdamForwardGradientEstimator(AbstractGradientEstimator):
    def __init__(
        self,
        parameters: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        k_update_strategy: KUpdateStrategy = KUpdateStrategy.LAST_LOCAL_UPDATE,
        hessian_smooth: float = 0.95,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(f"trainable model size: {self.total_dimensions}")

        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype

        self.k_update_strategy: KUpdateStrategy = k_update_strategy
        self.hessian_smooth = hessian_smooth  # test fine tune this
        self.K_vec = torch.ones(self.total_dimensions, device=self.device, dtype=self.torch_dtype)

    def construct_gradient(self, dir_grads: torch.Tensor, seed: int) -> torch.Tensor:
        update_grad: torch.Tensor | None = None
        num_pert = len(dir_grads)
        for i, dir_grad in enumerate(dir_grads):
            rng = self.get_rng(seed, i)
            update = self.generate_perturbation_norm(rng).mul_(dir_grad / num_pert)
            if update_grad is None:
                update_grad = update
            else:
                update_grad += update
        assert update_grad is not None
        return update_grad

    def update_K_vec(self, dir_grads: torch.Tensor, seed: int) -> None:
        grad = self.construct_gradient(dir_grads, seed)
        self.K_vec = self.hessian_smooth * self.K_vec + (1 - self.hessian_smooth) * grad.square_()

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(
            self.total_dimensions, device=self.device, dtype=self.torch_dtype, generator=rng
        ) / torch.sqrt(self.K_vec)

    def compute_grad(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        grad, perturbation_dir_grads = self._zo_grad_estimate(batch_inputs, labels, loss_fn, seed)
        self.put_grad(grad)
        return perturbation_dir_grads

    def _zo_grad_estimate(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_0 = loss_fn(batch_inputs, labels)
        grad: torch.Tensor | None = None
        dir_grads = []
        ## calculate dir_grads and dir_hessian_factors
        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)

            self.perturb_model(pb_norm, alpha=self.mu)
            loss_plus = loss_fn(batch_inputs, labels)
            self.perturb_model(pb_norm, alpha=-self.mu)

            dir_grad = (loss_plus - loss_0) / self.mu
            dir_grads.append(dir_grad)
            if grad is None:
                grad = dir_grad * pb_norm
            else:
                grad.add_(pb_norm, alpha=dir_grad)
            del pb_norm

        assert grad is not None

        return grad.div_(self.num_pert), torch.tensor(dir_grads, device=self.device)

    def update_gradient_estimator_given_seed_and_grad(
        self,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        """
        # seeds is [seed1, seed2, ...seedk, ] for K local updates
        # global_grad_scalar is [gradscalar1, gradscalar2, ...gradscalarK, ] for K local updates
        # strategy 1: update K k times for every k local updates
        # strategy 2: update K 1 time for every last local update
        # strategy 3: update K 1 time for average of K local updates. NOTE: this is not preferred
        """
        assert len(iteration_seeds) == len(iteration_grad_scalar)

        if self.k_update_strategy == KUpdateStrategy.LAST_LOCAL_UPDATE:
            self.update_K_vec(iteration_grad_scalar[-1], iteration_seeds[-1])
        elif self.k_update_strategy == KUpdateStrategy.ALL_LOCAL_UPDATES:
            for one_update_seed, one_update_grad_dirs in zip(
                iteration_seeds, iteration_grad_scalar
            ):
                self.update_K_vec(one_update_grad_dirs, one_update_seed)

    def update_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        assert len(iteration_seeds) == len(iteration_grad_scalar)

        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            # We don't really need optimizer.zero_grad() here because we put grad directly.
            self.generate_then_put_grad(one_update_seed, one_update_grad_dirs)
            # update model
            optimizer.step()
