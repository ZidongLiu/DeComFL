import torch
from typing import Sequence
from copy import deepcopy

from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from torch.optim import SGD
from cezo_fl.server import AbstractClient
from cezo_fl.shared import update_model_given_seed_and_grad


class Client(AbstractClient):

    def __init__(
        self,
        model,
        dataloader,
        device,
        grad_estimator,
        optimizer,
        criterion,
    ):
        self.model = model
        self.dataloader = dataloader

        self.device = device

        self.grad_estimator = grad_estimator
        self.optimizer = optimizer
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
        grad_scalar_list: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        # reset model
        self.reset_model()
        # update model to latest version
        update_model_given_seed_and_grad(
            self.model,
            self.optimizer,
            self.grad_estimator,
            seeds_list,
            grad_scalar_list,
        )

        # screenshot current pulled model
        self.last_pull_state_dict = self.screenshot()
