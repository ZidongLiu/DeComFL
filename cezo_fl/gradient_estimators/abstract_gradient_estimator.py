import torch

from typing import Callable, Sequence
from abc import ABC, abstractmethod


class AbstractGradientEstimator(ABC):
    num_pert: int
    device: str | torch.device | None
    torch_dtype: torch.dtype
    parameters_list: list[torch.nn.Parameter]
    sgd_only_no_optim: bool = False

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def perturb_model(self, perturb: torch.Tensor, alpha: float | int = 1) -> None:
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

    def generate_then_put_grad(self, seed: int, dir_grads: torch.Tensor) -> None:
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
        self.put_grad(update_grad)

    @abstractmethod
    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Generate a perturbation vector with the same dimension as the model parameters.
        This method should be implemented in subclasses. considering random variable can be uniform or normal.
        Also generated random vector can be modified by some scaling vector in decomfl v2 algorithms
        """
        pass

    @abstractmethod
    def compute_grad(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        """
        Compute the gradient of the model parameters with respect to the loss function.
        Petrurb model then calculate loss for that direction.
        Args:
            batch_inputs: input data for the model.
            labels: true labels for the input data.
            loss_fn: a callable that takes the model output and labels as input and returns the loss.
            seed: random seed for generating perturbations.
        Returns:
            dir_grads: torch.Tensor([dir_grad1, dir_grad2, ...]) where dir_grad1, dir_grad2, ... are the gradient scalar for that perturbation direction.
                    shape of dir_grads is number of perturbations
        """
        pass

    @abstractmethod
    def update_gradient_estimator_given_seed_and_grad(
        self,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        """
        Update the gradient_estimator helpers(for decomfl v2 algorithms, like K_vec, smoothing_factor) using seeds and associated perturbation gradients.
        seeds is [seed1, seed2, ...seedk, ] for K local updates
        global_grad_scalar is [gradscalar1, gradscalar2, ...gradscalarK, ] for K local updates
        strategy 1: update K k times for every k local updates
        strategy 2: update K 1 time for every last local update
        strategy 3: update K 1 time for average of K local updates. NOTE: this is not preferred
        Args:
            iteration_seeds: list of seeds for 1 iteration in decomfl framework.
            iteration_grad_scalar: list of gradient scalars for 1 iteration in decomfl framework.
        """
        pass

    @abstractmethod
    def update_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        """
        Update the model parameters using the optimizer and the provided gradients.
        Args:
            optimizer: optimizer to update the model parameters.
            iteration_seeds: list of seeds for 1 iteration in decomfl framework.
            iteration_grad_scalar: list of gradient scalars for 1 iteration in decomfl framework.
        """
        pass
