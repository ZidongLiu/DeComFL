import torch
from typing import Sequence
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE


def update_model_given_seed_and_grad(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_estimator: RGE,
    seeds_list: Sequence[Sequence[int]],
    grad_scalar_list: Sequence[Sequence[torch.Tensor]],
) -> None:
    assert len(seeds_list) == len(grad_scalar_list)
    for iteration_seeds, iteration_grad_avgs in zip(seeds_list, grad_scalar_list):
        for local_update_seed, local_update_grad_vector in zip(
            iteration_seeds, iteration_grad_avgs
        ):
            # create gradient
            torch.manual_seed(local_update_seed)
            this_num_pert = local_update_grad_vector.shape[0]
            perturbations = [
                grad_estimator.generate_perturbation_norm()
                for _ in range(this_num_pert)
            ]

            update_grad = (
                sum(
                    [
                        perturb * local_update_grad_vector[j]
                        for j, perturb in enumerate(perturbations)
                    ]
                )
                / this_num_pert
            )
            grad_estimator.put_grad(update_grad)
            # update model
            optimizer.step()
    return model
