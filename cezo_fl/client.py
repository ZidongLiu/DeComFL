import torch
from torch.utils.data import DataLoader
from typing import Sequence
from copy import deepcopy

from shared.metrics import Metric
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.server import AbstractClient, LocalUpdateResult
from cezo_fl.shared import (
    CriterionType,
    update_model_given_seed_and_grad,
    revert_SGD_given_seed_and_grad,
)


class SyncClient(AbstractClient):

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        grad_estimator: RGE,
        optimizer: torch.optim.Optimizer,
        criterion: CriterionType,
        accuracy_func,
        device: str | None = None,
    ):
        self.model = model
        self.dataloader = dataloader

        self.device = device

        self.grad_estimator = grad_estimator
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.data_iterator = self._get_train_batch_iterator()

        self.local_update_seeds: list[int] = []
        self.local_update_dir_grads: list[torch.Tensor] = []

    def random_gradient_estimator(self):
        return self.grad_estimator

    def _get_train_batch_iterator(self):
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        iteration_local_update_grad_vectors: Sequence[torch.Tensor] = []
        train_loss = Metric("Client train loss")
        train_accuracy = Metric("Client train accuracy")

        for seed in seeds:
            self.optimizer.zero_grad()
            # NOTE:dataloader manage its own randomnes state thus not affected by seed
            batch_inputs, labels = next(self.data_iterator)
            if self.device != torch.device("cpu") or self.grad_estimator.torch_dtype != torch.float32:
                batch_inputs, labels = batch_inputs.to(self.device, self.grad_estimator.torch_dtype), labels.to(self.device, self.grad_estimator.torch_dtype)
            # generate grads and update model's gradient
            torch.manual_seed(seed)
            seed_grads = self.grad_estimator.compute_grad(batch_inputs, labels, self.criterion)
            iteration_local_update_grad_vectors.append(seed_grads)

            # update model
            # NOTE: local model update also uses momentum and other states
            self.optimizer.step()

            # get_train_info
            pred = self.grad_estimator.model_forward(batch_inputs)
            train_loss.update(self.criterion(pred, labels))
            train_accuracy.update(self.accuracy_func(pred, labels))

        # This should only run 1 time before next pull, but still use append instead of assign to
        # prevent potential bug
        self.local_update_seeds += seeds
        self.local_update_dir_grads += iteration_local_update_grad_vectors

        return LocalUpdateResult(
            grad_tensors=iteration_local_update_grad_vectors,
            step_accuracy=train_accuracy.avg,
            step_loss=train_loss.avg,
        )

    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        assert isinstance(self.optimizer, torch.optim.SGD)
        revert_SGD_given_seed_and_grad(
            self.optimizer,
            self.grad_estimator,
            self.local_update_seeds,
            self.local_update_dir_grads,
        )

    def screenshot(self) -> None:
        # deepcopy current model.state_dict and optimizer.state_dict
        self.last_pull_state_dict = deepcopy({"optimizer": self.optimizer.state_dict()})
        self.local_update_seeds_list = []
        self.local_update_grad_scalar = []

    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        # reset model
        self.reset_model()
        # update model to latest version
        for iteration_seeds, iteration_grad_sclar in zip(seeds_list, gradient_scalar):
            update_model_given_seed_and_grad(
                self.optimizer,
                self.grad_estimator,
                iteration_seeds,
                iteration_grad_sclar,
            )

        # screenshot current pulled model
        self.screenshot()


class ResetClient(AbstractClient):

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        grad_estimator: RGE,
        optimizer: torch.optim.Optimizer,
        criterion: CriterionType,
        accuracy_func,
        device: str | None = None,
    ):
        self.model = model
        self.dataloader = dataloader

        self.device = device

        self.grad_estimator = grad_estimator
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.data_iterator = self._get_train_batch_iterator()
        self.last_pull_state_dict = self.screenshot()

    def random_gradient_estimator(self):
        return self.grad_estimator

    def _get_train_batch_iterator(self):
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        iteration_local_update_grad_vectors: Sequence[torch.Tensor] = []
        train_loss = Metric("Client train loss")
        train_accuracy = Metric("Client train accuracy")

        for seed in seeds:
            self.optimizer.zero_grad()
            # NOTE:dataloader manage its own randomnes state thus not affected by seed
            batch_inputs, labels = next(self.data_iterator)
            if self.device != torch.device("cpu"):
                batch_inputs, labels = batch_inputs.to(self.device), labels.to(self.device)
            # generate grads and update model's gradient
            torch.manual_seed(seed)
            seed_grads = self.grad_estimator.compute_grad(batch_inputs, labels, self.criterion)
            iteration_local_update_grad_vectors.append(seed_grads)

            # update model
            # NOTE: local model update also uses momentum and other states
            self.optimizer.step()

            # get_train_info
            pred = self.grad_estimator.model_forward(batch_inputs)
            train_loss.update(self.criterion(pred, labels))
            train_accuracy.update(self.accuracy_func(pred, labels))

        return LocalUpdateResult(
            grad_tensors=iteration_local_update_grad_vectors,
            step_accuracy=train_accuracy.avg,
            step_loss=train_loss.avg,
        )

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
        for iteration_seeds, iteration_grad_sclar in zip(seeds_list, gradient_scalar):
            update_model_given_seed_and_grad(
                self.optimizer,
                self.grad_estimator,
                iteration_seeds,
                iteration_grad_sclar,
            )

        # screenshot current pulled model
        self.last_pull_state_dict = self.screenshot()
