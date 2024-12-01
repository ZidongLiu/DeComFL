from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Sequence

import torch

from cezo_fl.shared import CriterionType
from cezo_fl.util.metrics import Metric

from fed_avg.client import FedAvgClient


class FedAvgServer:
    def __init__(
        self,
        clients: Sequence[FedAvgClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_model_inference: Callable,
        server_criterion: CriterionType,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.server_model = server_model
        self.server_model_inference = server_model_inference
        self.server_criterion = server_criterion
        self.server_accuracy_func = server_accuracy_func

        self.dtype = next(server_model.parameters()).dtype

    def get_sampled_client_index(self) -> list[int]:
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def aggregate_client_models(self, client_indices: list[int]) -> None:
        self.server_model.train()
        with torch.no_grad():
            running_sum: Sequence[torch.Tensor] = [0.0 for _ in self.server_model.parameters()]  # type: ignore[misc, use 0 to start calculcation for tensor]
            for client_index in client_indices:
                client = self.clients[client_index]
                for i, p in enumerate(client.model.parameters()):
                    running_sum[i] += p.to(self.device)

            for model_p, to_set_p in zip(self.server_model.parameters(), running_sum):
                temp = to_set_p.div_(self.num_sample_clients)
                model_p.set_(temp)  # type: ignore[call-overload, this method takes Tensor as input but not allowed here, pytorch typing is off]

    def train_one_step(self) -> tuple[float, float]:
        # Step 0: initiate something
        sampled_client_indices: list[int] = self.get_sampled_client_index()

        # Step 1 & 2: pull model and local update
        step_train_loss = Metric("train_loss")
        step_train_accuracy = Metric("train_loss")
        for index in sampled_client_indices:
            client = self.clients[index]
            client.pull_model(self.server_model)
            client_loss, client_accuracy = client.local_update(self.local_update_steps)
            step_train_loss.update(client_loss)
            step_train_accuracy.update(client_accuracy)

        self.aggregate_client_models(sampled_client_indices)

        return step_train_loss.avg, step_train_accuracy.avg

    def eval_model(self, test_loader: Iterable[Any], iteration: int) -> tuple[float, float]:
        self.server_model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if self.device != torch.device("cpu") or self.dtype != torch.float32:
                    batch_inputs = batch_inputs.to(self.device, self.dtype)
                    # NOTE: label does not convert to dtype
                    batch_labels = batch_labels.to(self.device)
                pred = self.server_model_inference(self.server_model, batch_inputs)
                eval_loss.update(self.server_criterion(pred, batch_labels))
                eval_accuracy.update(self.server_accuracy_func(pred, batch_labels))
        print(
            f"\nEvaluation(Iteration {iteration + 1}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
