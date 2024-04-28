import torch
from typing import Sequence
from copy import deepcopy

from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from torch.optim import SGD
from cezo_fl.server import AbstractClient


class Client(AbstractClient):

    def __init__(
        self,
        model,
        dataloader,
        device,
        grad_estimator_params: dict,
        optimzier_params: dict,
        criterion,
    ):
        self.model = model
        self.dataloader = dataloader

        self.device = device

        self.grad_estimator = self._initialize_grad_estimator(grad_estimator_params)

        self.optimizer = self._initialize_optimizer(optimzier_params)
        self.criterion = criterion

        self.data_iterator = self._get_train_batch_iterator()
        self.last_pull_state_dict = self.screenshot()

    def _initialize_grad_estimator(self, grad_estimator_params):
        if grad_estimator_params["method"] in ["rge-forward", "rge-central"]:
            method = grad_estimator_params["method"][4:]

            return RGE(
                self.model,
                grad_estimate_method=method,
                mu=grad_estimator_params["mu"],
                num_pert=grad_estimator_params["num_pert"],
                device=self.device,
            )
        else:
            raise Exception(f'{grad_estimator_params["method"]} is not supported')

    def _initialize_optimizer(self, optimzier_params):
        if optimzier_params["method"] == "SGD":
            return SGD(
                self.model.parameters(),
                lr=optimzier_params["lr"],
                momentum=optimzier_params["momentum"],
                weight_decay=1e-5,
            )
        else:
            raise Exception(f'{optimzier_params["method"]} is not supported')

    def _get_train_batch_iterator(self):
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def local_update(self, seeds: Sequence[int]) -> Sequence[torch.Tensor]:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        ret: Sequence[torch.Tensor] = []
        for seed in seeds:
            # NOTE:dataloader manage its own randomnes state thus not affected by seed
            batch_inputs, labels = next(self.data_iterator)
            if self.device != torch.device("cpu"):
                batch_inputs, labels = batch_inputs.to(self.device), labels.to(
                    self.device
                )
            # generate grads and update model's gradient
            torch.manual_seed(seed)
            seed_grads = self.grad_estimator.compute_grad(
                batch_inputs, labels, self.criterion
            )
            ret.append(seed_grads)

            # update model
            # NOTE: local model update also uses momentum and other states
            self.optimizer.step()

        return ret

    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        self.model.load_state_dict(self.last_pull_state_dict["model"])
        self.optimizer.load_state_dict(self.last_pull_state_dict["optimizer"])

    def screenshot(self) -> dict:
        # deepcopy current model.state_dict and optimizer.state_dict
        return deepcopy(
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        )

    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        # reset model
        self.reset_model()
        # update model to latest version
        for iteration_seeds, iteration_grad_avgs in zip(seeds_list, gradient_scalar):
            for local_update_seed, local_update_grad_vector in zip(
                iteration_seeds, iteration_grad_avgs
            ):
                # create gradient
                torch.manual_seed(local_update_seed)
                this_num_pert = local_update_grad_vector.shape[0]
                perturbations = [
                    self.grad_estimator.generate_perturbation_norm()
                    for _ in range(this_num_pert)
                ]
                # NOTE: this following code only works when local_update_grad_vector is vector
                update_grad = (
                    sum(
                        [
                            perturb * local_update_grad_vector[j]
                            for j, perturb in enumerate(perturbations)
                        ]
                    )
                    / this_num_pert
                )
                self.grad_estimator.put_grad(update_grad)
                # update model
                self.optimizer.step()

        # screenshot current pulled model
        self.last_pull_state_dict = self.screenshot()
