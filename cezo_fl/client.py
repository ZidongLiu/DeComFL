from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Sequence, Callable, Any

import torch
from torch.utils.data import DataLoader

from cezo_fl.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradientEstimator
from cezo_fl.typing import CriterionType
from cezo_fl.util.metrics import Metric


@dataclass
class LocalUpdateResult:
    grad_tensors: list[torch.Tensor]
    step_accuracy: float
    step_loss: float

    # Must add __future__ import to be able to return, see https://stackoverflow.com/a/33533514
    def to(self, device: torch.device) -> LocalUpdateResult:
        self.grad_tensors = [grad_tensor.to(device) for grad_tensor in self.grad_tensors]
        return self


class AbstractClient:
    device: torch.device
    optimizer: torch.optim.Optimizer

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        return NotImplemented

    @abc.abstractmethod
    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        ...

    @abc.abstractmethod
    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        """Pull model from server side using seed and grad"""
        ...

    @abc.abstractmethod
    def gradient_estimator(self) -> AbstractGradientEstimator:
        return NotImplemented


class ResetClient(AbstractClient):
    def __init__(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        dataloader: DataLoader,
        grad_estimator: AbstractGradientEstimator,
        optimizer: torch.optim.Optimizer,
        criterion: CriterionType,
        accuracy_func,
        device: torch.device,
    ):
        self.model = model
        self.model_inference = model_inference
        self.dataloader = dataloader

        self.device = device

        self.grad_estimator = grad_estimator
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.data_iterator = self._get_train_batch_iterator()
        self.last_pull_state_dict: dict | None = self.screenshot()

    def gradient_estimator(self) -> AbstractGradientEstimator:
        return self.grad_estimator

    def _get_train_batch_iterator(self) -> Iterator:
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def _loss_fn(self, batch_inputs, batch_labels):
        return self.criterion(self.model_inference(self.model, batch_inputs), batch_labels)

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        iteration_local_update_grad_vectors: list[torch.Tensor] = []
        train_loss = Metric("Client train loss")
        train_accuracy = Metric("Client train accuracy")

        for seed in seeds:  # Length of seeds equals the number of local update.
            self.optimizer.zero_grad()
            # NOTE:dataloader manage its own randomnes state thus not affected by seed
            batch_inputs, labels = next(self.data_iterator)
            if (
                self.device != torch.device("cpu")
                or self.grad_estimator.torch_dtype != torch.float32
            ):
                batch_inputs = batch_inputs.to(self.device, self.grad_estimator.torch_dtype)
                if isinstance(labels, torch.Tensor):  # In generation mode, labels are not tensor.
                    labels = labels.to(self.device)  # NOTE: label does not convert to dtype

            # declare grad_scalars before assigning it to avoid no-redef type check
            grad_scalars: torch.Tensor
            if self.grad_estimator.sgd_only_no_optim and isinstance(
                self.grad_estimator, RandomGradientEstimator
            ):
                grad_scalars = self.grad_estimator._zo_grad_estimate_paramwise(
                    batch_inputs, labels, self._loss_fn, seed
                )
                self.grad_estimator.update_model_given_seed_and_grad(
                    self.optimizer, [seed], [grad_scalars]
                )
            else:
                # generate grads and update model's gradient
                # The length of grad_scalars is number of perturbations
                grad_scalars = self.grad_estimator.compute_grad(
                    batch_inputs, labels, self._loss_fn, seed
                )
                self.optimizer.step()
            iteration_local_update_grad_vectors.append(grad_scalars)

            # get_train_info
            pred = self.model_inference(self.model, batch_inputs)
            train_loss.update(self.criterion(pred, labels))
            train_accuracy.update(self.accuracy_func(pred, labels))

        return LocalUpdateResult(
            grad_tensors=iteration_local_update_grad_vectors,
            step_accuracy=train_accuracy.avg,
            step_loss=train_loss.avg,
        )

    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        assert self.last_pull_state_dict is not None
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
            self.grad_estimator.update_model_given_seed_and_grad(
                self.optimizer,
                iteration_seeds,
                iteration_grad_sclar,
            )
            self.grad_estimator.update_gradient_estimator_given_seed_and_grad(
                iteration_seeds,
                iteration_grad_sclar,
            )

        # screenshot current pulled model
        self.last_pull_state_dict = None  # remove previous record to avoid memory spike
        self.last_pull_state_dict = self.screenshot()


# class SyncClient(AbstractClient):
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
#         dataloader: DataLoader,
#         grad_estimator: AbstractGradientEstimator,
#         optimizer: torch.optim.Optimizer,
#         criterion: CriterionType,
#         accuracy_func,
#         device: torch.device,
#     ):
#         self.model = model
#         self.model_inference = model_inference
#         self.dataloader = dataloader

#         self.device = device

#         self.grad_estimator = grad_estimator
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.accuracy_func = accuracy_func

#         self.data_iterator = self._get_train_batch_iterator()

#         self.local_update_seeds: list[int] = []
#         self.local_update_dir_grads: list[torch.Tensor] = []

#     def _get_train_batch_iterator(self) -> Iterator:
#         # NOTE: used only in init, will generate an infinite iterator from dataloader
#         while True:
#             for v in self.dataloader:
#                 yield v

#     def _loss_fn(self, batch_inputs, batch_labels):
#         return self.criterion(self.model_inference(self.model, batch_inputs), batch_labels)

#     def random_gradient_estimator(self) -> AbstractGradientEstimator:
#         return self.grad_estimator

#     def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
#         """Returns a sequence of gradient scalar tensors for each local update.

#         The length of the returned sequence should be the same as the length of seeds.
#         The inner tensor can be a scalar or a vector. The length of vector is the number
#         of perturbations.
#         """
#         iteration_local_update_grad_vectors: list[torch.Tensor] = []
#         train_loss = Metric("Client train loss")
#         train_accuracy = Metric("Client train accuracy")

#         for seed in seeds:  # Length of seeds equals the number of local update.
#             self.optimizer.zero_grad()
#             # NOTE:dataloader manage its own randomnes state thus not affected by seed
#             batch_inputs, labels = next(self.data_iterator)
#             if (
#                 self.device != torch.device("cpu")
#                 or self.grad_estimator.torch_dtype != torch.float32
#             ):
#                 batch_inputs = batch_inputs.to(self.device, self.grad_estimator.torch_dtype)
#                 if isinstance(labels, torch.Tensor):  # In generation mode, labels are not tensor.
#                     labels = labels.to(self.device)  # NOTE: label does not convert to dtype

#             # declare grad_scalars before assigning it to avoid no-redef type check
#             grad_scalars: torch.Tensor
#             if self.grad_estimator.sgd_only_no_optim:
#                 grad_scalars = self.grad_estimator._zo_grad_estimate_paramwise(
#                     batch_inputs, labels, self._loss_fn, seed
#                 )
#                 self.grad_estimator.update_model_given_seed_and_grad(
#                     self.optimizer, [seed], [grad_scalars]
#                 )
#             else:
#                 # generate grads and update model's gradient
#                 # The length of grad_scalars is number of perturbations
#                 grad_scalars = self.grad_estimator.compute_grad(
#                     batch_inputs, labels, self._loss_fn, seed
#                 )
#                 self.optimizer.step()
#             iteration_local_update_grad_vectors.append(grad_scalars)

#             # get_train_info
#             pred = self.model_inference(self.model, batch_inputs)
#             train_loss.update(self.criterion(pred, labels))
#             train_accuracy.update(self.accuracy_func(pred, labels))

#         # This should only run 1 time before next pull, but still use append instead of assign to
#         # prevent potential bug
#         self.local_update_seeds += seeds
#         self.local_update_dir_grads += iteration_local_update_grad_vectors

#         return LocalUpdateResult(
#             grad_tensors=iteration_local_update_grad_vectors,
#             step_accuracy=train_accuracy.avg,
#             step_loss=train_loss.avg,
#         )

#     def reset_model(self) -> None:
#         """Reset the mode to the state before the local_update."""
#         assert isinstance(self.optimizer, torch.optim.SGD)
#         self.grad_estimator.revert_model_given_seed_and_grad(
#             self.optimizer,
#             self.local_update_seeds,
#             self.local_update_dir_grads,
#         )

#     def screenshot(self) -> None:
#         # deepcopy current model.state_dict and optimizer.state_dict
#         self.last_pull_state_dict = deepcopy({"optimizer": self.optimizer.state_dict()})
#         self.local_update_seeds_list: list[int] = []
#         self.local_update_grad_scalar: list[torch.Tensor] = []

#     def pull_model(
#         self,
#         seeds_list: Sequence[Sequence[int]],
#         gradient_scalar: Sequence[Sequence[torch.Tensor]],
#     ) -> None:
#         # reset model
#         self.reset_model()
#         # update model to latest version
#         for iteration_seeds, iteration_grad_sclar in zip(seeds_list, gradient_scalar):
#             self.grad_estimator.update_model_given_seed_and_grad(
#                 self.optimizer,
#                 iteration_seeds,
#                 iteration_grad_sclar,
#             )

#         # screenshot current pulled model
#         self.screenshot()
