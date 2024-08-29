import torch
from typing import Sequence, Callable, TypeAlias
from gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator as RGE,
)

CriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_update_grad_for_1_seed(grad_estimator: RGE, perturb_grad_vector: torch.Tensor, seed: int):
    # 1 seed is in charge all perturbs
    rng = torch.Generator(device=grad_estimator.device).manual_seed(seed)
    update_grad = None
    for local_update_grad in perturb_grad_vector:
        perturb = grad_estimator.generate_perturbation_norm(rng)
        if update_grad is None:
            update_grad = perturb.mul_(local_update_grad)
        else:
            # TODO: fix type here (Do we really have type issue here???)
            update_grad.add_(perturb, alpha=local_update_grad)
        del perturb

    assert isinstance(update_grad, torch.Tensor)
    return update_grad.div_(perturb_grad_vector.shape[0])


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
        update_grad = get_update_grad_for_1_seed(
            grad_estimator, local_update_grad_vector, local_update_seed
        )
        grad_estimator.put_grad(update_grad)
        # update model
        optimizer.step()


def revert_SGD_given_seed_and_grad(
    optimizer: torch.optim.SGD,
    grad_estimator: RGE,
    iteration_seeds: Sequence[int],
    iteration_grad_scalar: Sequence[torch.Tensor],  # this should be stored in each client
) -> None:
    """
    This only works with SGD without momentum and without lr scheduling
    """
    try:
        assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults["momentum"] == 0
    except AssertionError:
        raise Exception("Revert only supports SGD without momentum")

    lr, weight_decay = optimizer.defaults["lr"], optimizer.defaults["weight_decay"]

    n_update = len(iteration_seeds)
    # reverse loop the seed and scalar
    for i in reversed(range(n_update)):
        local_update_seed, local_update_grad_vector = iteration_seeds[i], iteration_grad_scalar[i]
        # create gradient
        update_grad = get_update_grad_for_1_seed(
            grad_estimator, local_update_grad_vector, local_update_seed
        )

        # update model
        # 1. update using gradient
        grad_estimator.perturb_model(update_grad, lr)
        # 2. scale down with weight_decay
        if weight_decay > 0:
            grad_estimator.perturb_model(perturb=None, alpha=1 / (1 - lr * weight_decay))
