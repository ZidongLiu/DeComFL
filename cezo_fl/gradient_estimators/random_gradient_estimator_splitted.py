from typing import Callable, Iterator, Sequence

import torch
from torch.nn import Parameter
from cezo_fl.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradEstimateMethod


class RandomGradientEstimatorBatch(AbstractGradientEstimator):
    def __init__(
        self,
        parameters: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        grad_estimate_method: RandomGradEstimateMethod | str = RandomGradEstimateMethod.rge_central,
        normalize_perturbation: bool = False,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(f"Using RandomGradientEstimatorBatch, trainable model size: {self.total_dimensions}")

        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype
        if isinstance(grad_estimate_method, RandomGradEstimateMethod):
            self.grad_estimate_method: RandomGradEstimateMethod = grad_estimate_method
        else:
            if grad_estimate_method == RandomGradEstimateMethod.rge_central.value:
                self.grad_estimate_method = RandomGradEstimateMethod.rge_central
            elif grad_estimate_method == RandomGradEstimateMethod.rge_forward.value:
                self.grad_estimate_method = RandomGradEstimateMethod.rge_forward
            else:
                raise Exception("Grad estimate method has to be rge-central or rge-forward")

        self.normalize_perturbation = normalize_perturbation

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        p = torch.randn(
            self.total_dimensions, device=self.device, dtype=self.torch_dtype, generator=rng
        )

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

    def compute_grad(self, batch_inputs, labels, loss_fn, seed: int) -> torch.Tensor:
        # We generate the perturbation vector all together. It should be faster but consume
        # more memory
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
        """Calculate the zeroth-order gradient estimate.

        Return a tuple, the first element is full grad and the second is the gradient scalar.

           g_full = avg_{p} (g_p*z_p),
           where  g_p = [loss(x+mu*z_p) - loss(x)] / mu ------------------- forward approach
                  g_p = [loss(x+mu*z_p) - loss(x-mu*z_p)] / (2*mu) -------- central approach

        i.e., returning (g_full, [g_1, g_2, ..., g_p]).
        """
        grad: torch.Tensor | None = None
        dir_grads = []
        denominator_factor = (
            2 if self.grad_estimate_method == RandomGradEstimateMethod.rge_central else 1
        )
        if self.grad_estimate_method == RandomGradEstimateMethod.rge_forward:
            pert_minus_loss = loss_fn(batch_inputs, labels)

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)

            self.perturb_model(pb_norm, alpha=self.mu)
            pert_plus_loss = loss_fn(batch_inputs, labels)
            if self.grad_estimate_method == RandomGradEstimateMethod.rge_central:
                self.perturb_model(pb_norm, alpha=-2 * self.mu)
                pert_minus_loss = loss_fn(batch_inputs, labels)
                self.perturb_model(pb_norm, alpha=self.mu)  # Restore model
            elif self.grad_estimate_method == RandomGradEstimateMethod.rge_forward:
                self.perturb_model(pb_norm, alpha=-self.mu)  # Restore model

            dir_grad = (pert_plus_loss - pert_minus_loss) / (self.mu * denominator_factor)
            dir_grads += [dir_grad]
            if grad is None:
                grad = pb_norm.mul_(dir_grad)
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
        # No updates needed for this class.
        pass

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

    def revert_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        assert len(iteration_seeds) == len(iteration_grad_scalar)
        try:
            assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults["momentum"] == 0
        except AssertionError:
            raise Exception("Revert only supports SGD without momentum")

        lr, weight_decay = optimizer.defaults["lr"], optimizer.defaults["weight_decay"]
        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            # We don't really need optimizer.zero_grad() here because we put grad directly.
            self.generate_then_put_grad(one_update_seed, one_update_grad_dirs)

            for param in self.parameters_list:
                assert param.grad is not None
                param.add_(param.grad, alpha=lr)  # gradient ascent instead of descent.
                if weight_decay > 0:
                    param.mul_(1 / (1 - lr * weight_decay))


class RandomGradientEstimatorParamwise(AbstractGradientEstimator):
    def __init__(
        self,
        parameters: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        grad_estimate_method: RandomGradEstimateMethod | str = RandomGradEstimateMethod.rge_central,
        normalize_perturbation: bool = False,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(
            f"Using RandomGradientEstimatorParamwise, trainable model size: {self.total_dimensions}"
        )

        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype
        if isinstance(grad_estimate_method, RandomGradEstimateMethod):
            self.grad_estimate_method: RandomGradEstimateMethod = grad_estimate_method
        else:
            if grad_estimate_method == RandomGradEstimateMethod.rge_central.value:
                self.grad_estimate_method = RandomGradEstimateMethod.rge_central
            elif grad_estimate_method == RandomGradEstimateMethod.rge_forward.value:
                self.grad_estimate_method = RandomGradEstimateMethod.rge_forward
            else:
                raise Exception("Grad estimate method has to be rge-central or rge-forward")

        self.normalize_perturbation = normalize_perturbation

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError("This method should not be called for this class")

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def compute_grad(self, batch_inputs, labels, loss_fn, seed: int) -> torch.Tensor:
        perturbation_dir_grads = self._zo_grad_estimate_paramwise(
            batch_inputs, labels, loss_fn, seed
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
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        dir_grads = []
        denominator_factor = (
            2 if self.grad_estimate_method == RandomGradEstimateMethod.rge_central else 1
        )
        if self.grad_estimate_method == RandomGradEstimateMethod.rge_forward:
            pert_minus_loss = loss_fn(batch_inputs, labels)

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            self.perturb_model_paramwise(rng, alpha=self.mu)
            pert_plus_loss = loss_fn(batch_inputs, labels)
            if self.grad_estimate_method == RandomGradEstimateMethod.rge_central:
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=-2 * self.mu)
                pert_minus_loss = loss_fn(batch_inputs, labels)
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=self.mu)  # Restore model
            elif self.grad_estimate_method == RandomGradEstimateMethod.rge_forward:
                rng = self.get_rng(seed, i)
                self.perturb_model_paramwise(rng, alpha=-self.mu)  # Restore model
            dir_grad = (pert_plus_loss - pert_minus_loss) / (self.mu * denominator_factor)
            dir_grads += [dir_grad]
        return torch.tensor(dir_grads, device=self.device)

    def update_gradient_estimator_given_seed_and_grad(
        self,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        # No updates needed for this class.
        pass

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

    def revert_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        assert len(iteration_seeds) == len(iteration_grad_scalar)
        try:
            assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults["momentum"] == 0
        except AssertionError:
            raise Exception("Revert only supports SGD without momentum")

        lr, weight_decay = optimizer.defaults["lr"], optimizer.defaults["weight_decay"]
        for one_update_seed, one_update_grad_dirs in zip(iteration_seeds, iteration_grad_scalar):
            # We don't really need optimizer.zero_grad() here because we put grad directly.
            self.generate_then_put_grad_paramwise(one_update_seed, one_update_grad_dirs)

            for param in self.parameters_list:
                assert param.grad is not None
                param.add_(param.grad, alpha=lr)  # gradient ascent instead of descent.
                if weight_decay > 0:
                    param.mul_(1 / (1 - lr * weight_decay))
