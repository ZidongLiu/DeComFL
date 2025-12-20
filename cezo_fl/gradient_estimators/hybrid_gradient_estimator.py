from typing import Callable, Iterator, Sequence
from cezo_fl.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator
from cezo_fl.gradient_estimators.adam_forward import KUpdateStrategy

import torch
from torch.nn import Parameter


class HybridGradientEstimatorBatch(AbstractGradientEstimator):
    def __init__(
        self,
        random_parameters_list: Iterator[Parameter],
        adam_forward_parameters_list: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        k_update_strategy: KUpdateStrategy = KUpdateStrategy.LAST_LOCAL_UPDATE,
        hessian_smooth: float = 0.95,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.random_parameters_list: list[Parameter] = [
            p for p in random_parameters_list if p.requires_grad
        ]
        self.adam_forward_parameters_list: list[Parameter] = [
            p for p in adam_forward_parameters_list if p.requires_grad
        ]
        self.parameters_list = self.random_parameters_list + self.adam_forward_parameters_list

        self.random_gradient_dimensions = sum([p.numel() for p in self.random_parameters_list])
        self.adam_forward_dimensions = sum([p.numel() for p in self.adam_forward_parameters_list])
        self.total_dimensions = self.random_gradient_dimensions + self.adam_forward_dimensions
        print(
            f"Using HybridGradientEstimatorBatch, trainable model size: {self.total_dimensions}, random_gradient_dimensions: {self.random_gradient_dimensions}, adam_forward_dimensions: {self.adam_forward_dimensions}, total_dimensions: {self.total_dimensions}"
        )
        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype

        self.k_update_strategy: KUpdateStrategy = k_update_strategy
        self.hessian_smooth = hessian_smooth  # test fine tune this
        self.K_vec = torch.ones(
            self.adam_forward_dimensions, device=self.device, dtype=self.torch_dtype
        )

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
        self.K_vec = (
            self.hessian_smooth * self.K_vec
            + (1 - self.hessian_smooth) * grad[self.random_gradient_dimensions :].square_()
        )

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Generate a random perturbation vector. Concatenate the random gradient and the adam forward gradient.
        rng is only used in this method, thus the state should still be stable
        """
        random_gradient = torch.randn(
            self.random_gradient_dimensions,
            device=self.device,
            dtype=self.torch_dtype,
            generator=rng,
        )
        adam_forward_gradient = torch.randn(
            self.adam_forward_dimensions, device=self.device, dtype=self.torch_dtype, generator=rng
        ) / torch.sqrt(self.K_vec)
        return torch.cat([random_gradient, adam_forward_gradient], dim=0)

    def put_grad(self, grad: torch.Tensor) -> None:
        """
        grad is a tensor of shape (total_dimensions,), concatenate the random gradient and the adam forward gradient.
        Put the grad first into the random parameters and then the adam forward parameters.
        """
        start = 0
        for p in self.random_parameters_list:
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

        for p in self.adam_forward_parameters_list:
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

        assert start == self.total_dimensions

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


class HybridGradientEstimatorParamwise(AbstractGradientEstimator):
    def __init__(
        self,
        random_parameters_list: Iterator[Parameter],
        adam_forward_parameters_list: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        k_update_strategy: KUpdateStrategy = KUpdateStrategy.LAST_LOCAL_UPDATE,
        hessian_smooth: float = 0.95,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.random_parameters_list: list[Parameter] = [
            p for p in random_parameters_list if p.requires_grad
        ]
        self.adam_forward_parameters_list: list[Parameter] = [
            p for p in adam_forward_parameters_list if p.requires_grad
        ]
        self.parameters_list = self.random_parameters_list + self.adam_forward_parameters_list
        self.random_parameters_count = len(self.random_parameters_list)

        self.random_gradient_dimensions = sum([p.numel() for p in self.random_parameters_list])
        self.adam_forward_dimensions = sum([p.numel() for p in self.adam_forward_parameters_list])
        self.total_dimensions = self.random_gradient_dimensions + self.adam_forward_dimensions
        print(
            f"Using HybridGradientEstimatorParamwise, trainable model size: {self.total_dimensions}, random_gradient_dimensions: {self.random_gradient_dimensions}, adam_forward_dimensions: {self.adam_forward_dimensions}"
        )

        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype

        self.k_update_strategy: KUpdateStrategy = k_update_strategy
        self.hessian_smooth = hessian_smooth  # test fine tune this
        # Create a list of K_param tensors, one for each parameter with same shape
        self.K_param_list: list[torch.Tensor] = [
            torch.ones(p.shape, device=self.device, dtype=self.torch_dtype)
            for p in self.adam_forward_parameters_list
        ]

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError("This method should not be called for this class")

    def generate_perturbation_norm_paramwise(
        self, param_index: int, rng: torch.Generator
    ) -> torch.Tensor:
        param = self.parameters_list[param_index]
        if param_index < self.random_parameters_count:
            ## return a random gradient for the random parameter when index is less than the length of random parameters list
            return torch.randn(
                *param.shape, device=self.device, dtype=self.torch_dtype, generator=rng
            )

        k_idx = param_index - self.random_parameters_count
        param_k = self.K_param_list[k_idx]
        return torch.randn(
            *param.shape, device=self.device, dtype=self.torch_dtype, generator=rng
        ).div_(torch.sqrt(param_k))

    def compute_grad(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        perturbation_dir_grads = self._zo_grad_estimate_paramwise(
            batch_inputs, labels, loss_fn, seed
        )
        self.generate_then_put_grad_paramwise(seed, perturbation_dir_grads)
        return perturbation_dir_grads

    def _zo_grad_estimate_paramwise(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        loss_0 = loss_fn(batch_inputs, labels)
        dir_grads = []

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            self.perturb_model_paramwise(rng, alpha=self.mu)
            loss_plus = loss_fn(batch_inputs, labels)
            # need to reset rng after using it to be able to recover
            rng = self.get_rng(seed, i)
            self.perturb_model_paramwise(rng, alpha=-self.mu)

            dir_grad = (loss_plus - loss_0) / self.mu
            dir_grads.append(dir_grad)

        return torch.tensor(dir_grads, device=self.device)

    def perturb_model_paramwise(self, rng: torch.Generator, alpha: float | int) -> None:
        for param_idx, param in enumerate(self.parameters_list):
            _perturb = self.generate_perturbation_norm_paramwise(param_idx, rng)
            param.add_(_perturb, alpha=alpha)
            del _perturb

    def generate_then_put_grad_paramwise(self, seed: int, dir_grads: torch.Tensor) -> None:
        num_pert = len(dir_grads)
        for i, dir_grad in enumerate(dir_grads):
            rng = self.get_rng(seed, i)
            for param_idx, param in enumerate(self.parameters_list):
                _perturb = self.generate_perturbation_norm_paramwise(param_idx, rng)

                if i == 0:
                    param.grad = _perturb.mul_(dir_grad / num_pert)
                else:
                    param.grad += _perturb.mul_(dir_grad / num_pert)
                del _perturb

    def sgd_no_optim_update_model(
        self, perturbation_dir_grads: torch.Tensor, seed: int, lr: float
    ) -> None:
        num_pert = len(perturbation_dir_grads)
        for i, dir_grad in enumerate(perturbation_dir_grads):
            rng = self.get_rng(seed, i)
            for param_idx, param in enumerate(self.parameters_list):
                _perturb = self.generate_perturbation_norm_paramwise(param_idx, rng)
                param.data.add_(_perturb, alpha=-lr * float(dir_grad) / num_pert)
                del _perturb

    def update_K_param_paramwise(self, dir_grads: torch.Tensor, seed: int) -> None:
        # Update K_param_list for each parameter separately
        num_pert = len(dir_grads)

        for i, dir_grad in enumerate(dir_grads):
            rng = self.get_rng(seed, i)
            for param_idx, param in enumerate(self.parameters_list):
                k_idx = param_idx - self.random_parameters_count
                _perturb = self.generate_perturbation_norm_paramwise(param_idx, rng)
                if k_idx < 0:
                    continue

                # Update K_param for this parameter
                grad_squared = _perturb.square_() * (dir_grad / num_pert) ** 2
                self.K_param_list[k_idx] = (
                    self.hessian_smooth * self.K_param_list[k_idx]
                    + (1 - self.hessian_smooth) * grad_squared
                )
                del _perturb

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
            self.update_K_param_paramwise(iteration_grad_scalar[-1], iteration_seeds[-1])
        elif self.k_update_strategy == KUpdateStrategy.ALL_LOCAL_UPDATES:
            for one_update_seed, one_update_grad_dirs in zip(
                iteration_seeds, iteration_grad_scalar
            ):
                self.update_K_param_paramwise(one_update_grad_dirs, one_update_seed)

    def update_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        assert len(iteration_seeds) == len(iteration_grad_scalar)
        lr = optimizer.defaults["lr"]  # Assume only one parameter group with lr.
        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            self.sgd_no_optim_update_model(one_update_grad_dirs, one_update_seed, lr)
