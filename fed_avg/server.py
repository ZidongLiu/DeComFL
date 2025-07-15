from __future__ import annotations

from enum import Enum
import random
from typing import Any, Callable, Iterable, Sequence

import torch

from cezo_fl.typing import CriterionType
from cezo_fl.util.metrics import Metric

from fed_avg.client import FedAvgClient


class FOFLStrategy(Enum):
    fedavg = "fedavg"
    fedadagrad = "fedadagrad"
    fedyogi = "fedyogi"
    fedadam = "fedadam"


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
        fo_fl_strategy: FOFLStrategy = FOFLStrategy.fedavg,
        lr: float = 1e-3,
        fo_fl_beta1: float = 0.9,
        fo_fl_beta2: float = 0.999,
    ) -> None:
        print("server strategy", fo_fl_strategy)
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.server_model = server_model
        self.server_model_inference = server_model_inference
        self.server_criterion = server_criterion
        self.server_accuracy_func = server_accuracy_func

        self.dtype = next(server_model.parameters()).dtype

        self.lr = lr
        self.gamma = 1e-8
        self.fo_fl_beta1 = fo_fl_beta1
        self.fo_fl_beta2 = fo_fl_beta2

        self.ms: list[torch.Tensor] | None = None
        self.vs: list[torch.Tensor] | None = None
        self.fo_fl_strategy = fo_fl_strategy
        if fo_fl_strategy in [FOFLStrategy.fedadam, FOFLStrategy.fedyogi, FOFLStrategy.fedadagrad]:
            self.ms = [torch.zeros_like(p) for p in self.server_model.parameters()]
            self.vs = [torch.zeros_like(p) for p in self.server_model.parameters()]

    def get_sampled_client_index(self) -> list[int]:
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def aggregate_client_models(self, client_indices: list[int]) -> None:
        self.server_model.train()
        running_sum: Sequence[torch.Tensor] = [0.0 for _ in self.server_model.parameters()]  # type: ignore[misc, use 0 to start calculcation for tensor]

        if self.fo_fl_strategy == FOFLStrategy.fedavg:
            with torch.no_grad():
                for client_index in client_indices:
                    client = self.clients[client_index]
                    for i, p in enumerate(client.model.parameters()):
                        running_sum[i] += p.to(self.device)

                for i, (model_p, sum_p) in enumerate(
                    zip(self.server_model.parameters(), running_sum)
                ):
                    temp = sum_p.div_(self.num_sample_clients)
                    model_p.set_(temp)  # type: ignore[call-overload, this method takes Tensor as input but not allowed here, pytorch typing is off]

        elif self.fo_fl_strategy in [
            FOFLStrategy.fedadagrad,
            FOFLStrategy.fedyogi,
            FOFLStrategy.fedadam,
        ]:
            assert self.ms is not None and self.vs is not None
            with torch.no_grad():
                for client_index in client_indices:
                    client = self.clients[client_index]
                    for i, p in enumerate(client.model.parameters()):
                        running_sum[i] += p.to(self.device)

                for i, (model_p, sum_p) in enumerate(
                    zip(self.server_model.parameters(), running_sum)
                ):
                    delta_t = sum_p.div_(self.num_sample_clients) - model_p
                    self.ms[i].mul_(self.fo_fl_beta1).add_(delta_t, alpha=1 - self.fo_fl_beta1)
                    delta_t_squared = delta_t.pow(2)
                    if self.fo_fl_strategy == FOFLStrategy.fedadagrad:
                        self.vs[i].add_(delta_t_squared)
                    elif self.fo_fl_strategy == FOFLStrategy.fedyogi:
                        self.vs[i].add_(
                            delta_t_squared * torch.sign(self.vs[i] - delta_t_squared),
                            alpha=-(1 - self.fo_fl_beta2),
                        )
                    elif self.fo_fl_strategy == FOFLStrategy.fedadam:
                        self.vs[i].mul_(self.fo_fl_beta2).add_(
                            delta_t_squared, alpha=1 - self.fo_fl_beta2
                        )
                    model_p.add_(self.ms[i] / (self.vs[i] + self.gamma).sqrt(), alpha=self.lr)
        else:
            raise ValueError(f"Invalid FO-FL strategy: {self.fo_fl_strategy}")

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
