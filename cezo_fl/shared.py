import torch
from typing import Sequence
from gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator as RGE,
)


def update_model_given_seed_and_grad(
    optimizer: torch.optim.Optimizer,
    grad_estimator: RGE,
    iteration_seeds: Sequence[int],
    iteration_grad_scalar: Sequence[torch.Tensor],
) -> None:
    assert len(iteration_seeds) == len(iteration_grad_scalar)
    optimizer.zero_grad()
    for local_update_seed, local_update_grad_vector in zip(iteration_seeds, iteration_grad_scalar):
        # create gradient
        torch.manual_seed(local_update_seed)
        update_grad = 0
        for local_update_grad in local_update_grad_vector:
            perturb = grad_estimator.generate_perturbation_norm()
            update_grad += perturb * local_update_grad
        grad_estimator.put_grad(update_grad.div_(local_update_grad_vector.shape[0]))
        # update model
        optimizer.step()
