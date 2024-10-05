from typing import Iterator

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers.models.opt.modeling_opt import OPTForCausalLM

from cezo_fl.shared import CriterionType
from cezo_fl.util.metrics import Metric
from cezo_fl.util.model_helpers import model_forward


class FedAvgClient:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: CriterionType,
        accuracy_func,
        device: torch.device | None = None,
    ):
        self.model = model
        self.dataloader = dataloader

        self._device = device

        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.data_iterator = self._get_train_batch_iterator()
        self.dtype = next(model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _get_train_batch_iterator(self) -> Iterator:
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v

    def local_update(self, local_update_steps: int) -> tuple[float, float]:
        train_loss = Metric("Client train loss")
        train_accuracy = Metric("Client train accuracy")

        for _ in range(local_update_steps):
            self.optimizer.zero_grad()
            # NOTE:dataloader manage its own randomnes state thus not affected by seed
            batch_inputs, labels = next(self.data_iterator)
            if self.device != torch.device("cpu") or self.dtype != torch.float32:
                batch_inputs = batch_inputs.to(self.device, self.dtype)
                # NOTE: label does not convert to dtype
                labels = labels.to(self.device)

            pred = model_forward(self.model, batch_inputs)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()
            # get_train_info
            train_loss.update(loss.detach().item())
            train_accuracy.update(self.accuracy_func(pred, labels).detach().item())

        return train_loss.avg, train_accuracy.avg

    def pull_model(self, server_model: OPTForCausalLM | PeftModel | torch.nn.Module) -> None:
        with torch.no_grad():
            for p, updated_p in zip(self.model.parameters(), server_model.parameters()):
                p.set_(updated_p.to(self._device))
