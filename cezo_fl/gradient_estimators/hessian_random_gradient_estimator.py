from typing import Callable, Iterator


import torch
from torch.nn import Parameter


class HessianRandomGradientEstimator:
    def __init__(
        self,
        parameters: Iterator[Parameter],
        mu=1e-3,
        num_pert=1,
        normalize_perturbation: bool = False,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(f"trainable model size: {self.total_dimensions}")

        self.hessian_smooth = 1e-6
        self.mu = mu
        self.num_pert = num_pert
        self.device = device
        self.torch_dtype = torch_dtype
        self.normalize_perturbation = normalize_perturbation
        self.hessian_vec = torch.ones(self.total_dimensions, device=self.device)

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

    def compute_grad(self, batch_inputs, labels, loss_fn, seed: int):
        grad = self._zo_grad_estimate(batch_inputs, labels, loss_fn, seed)
        self.put_grad(grad)

    def _zo_grad_estimate(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        """Calculate the zeroth-order gradient estimate.

        Return a tuple, the first element is full grad and the second is the gradient scalar.

           g_full = avg_{p} (g_p*z_p),
           where  g_p = [loss(x+mu*z_p) - loss(x)] / mu ------------------- forward approach
                  g_p = [loss(x+mu*z_p) - loss(x-mu*z_p)] / (2*mu) -------- central approach

        i.e., returning (g_full, [g_1, g_2, ..., g_p]).
        """
        grad: torch.Tensor | None = None
        hessian: torch.Tensor | None = None

        dir_grads = []
        dir_hessian_factors = []

        loss_0 = loss_fn(batch_inputs, labels)

        ## calculate dir_grads and dir_hessian_factors
        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)

            self.perturb_model(pb_norm, alpha=self.mu)
            loss_plus = loss_fn(batch_inputs, labels)
            self.perturb_model(pb_norm, alpha=-2 * self.mu)
            loss_minus = loss_fn(batch_inputs, labels)
            self.perturb_model(pb_norm, alpha=self.mu)  # Restore model

            dir_grad = (loss_plus - loss_minus) / (self.mu * 2)
            dir_grads += [dir_grad]

            dir_hessian_factor = abs((loss_plus + loss_minus - 2 * loss_0) / (2 * self.mu**2))
            dir_hessian_factors += [dir_hessian_factor]
            del pb_norm

        # calculate current hessian
        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)
            dir_hessian_vec = pb_norm**2 / self.hessian_vec
            if hessian is None:
                hessian = dir_hessian_vec.mul_(dir_hessian_factors[i])
            else:
                hessian.add_(dir_hessian_vec, alpha=dir_hessian_factors[i])

        assert hessian is not None
        # get smoothed hessian
        self.hessian_vec.mul_(1 - self.hessian_smooth).add_(hessian, alpha=self.hessian_smooth)

        # calculate grad
        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            dir_grad = dir_grads[i]

            pb_norm = self.generate_perturbation_norm(rng)
            projected_norm = self.hessian_vec * pb_norm
            if grad is None:
                grad = projected_norm.mul_(dir_grad)
            else:
                grad.add_(projected_norm, alpha=dir_grad)
        assert grad is not None
        return grad.div_(self.num_pert)
