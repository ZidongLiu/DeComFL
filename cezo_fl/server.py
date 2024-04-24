import abc
import random
import torch
from typing import Any, Iterable, Sequence
from collections import deque

from dataclasses import dataclass
from shared.metrics import Metric, accuracy


class AbstractClient:

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> Sequence[torch.Tensor]:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        return NotImplemented

    @abc.abstractmethod
    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        return NotImplemented

    @abc.abstractmethod
    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        return NotImplemented


def update_model_given_seed_and_grad(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    seeds: Sequence[int],
    grad_scalar_list: Sequence[torch.Tensor],
    device=None,
) -> None:
    assert len(seeds) == len(grad_scalar_list)
    param_len = sum([p.numel() for p in model.parameters()])

    def generate_perturbation_norm() -> torch.Tensor:
        p = torch.randn(param_len, device=device)
        return p

    def put_grad(grad: torch.Tensor) -> None:
        start = 0
        for p in model.parameters():
            p.grad = grad[start : (start + p.numel())].view(p.shape)
            start += p.numel()

    with torch.no_grad():
        # K-local update
        for seed, grad in zip(seeds, grad_scalar_list):
            optim.zero_grad()
            torch.manual_seed(seed)
            put_grad(
                sum(g * generate_perturbation_norm() for g in grad)
            )  # For multiple perturb
            optim.step()
    return model


class SeedAndGradientRecords:
    def __init__(self):
        self.seed_records: deque[list[int]] = deque()
        self.grad_records: deque[list[torch.Tensor]] = deque()
        self.earliest_records = 0
        self.current_iteration = -1

    def add_records(self, seeds: list[int], grad: list[torch.Tensor]) -> int:
        self.current_iteration += 1
        self.seed_records.append(seeds)
        self.grad_records.append(grad)
        return self.current_iteration

    def remove_too_old(self, earliest_record_needs: int):
        if self.earliest_records >= earliest_record_needs:
            return  # No need to do anything
        while self.earliest_records < earliest_record_needs:
            self.seed_records.popleft()
            self.grad_records.popleft()
            self.earliest_records += 1

    def fetch_seed_records(self, earliest_record_needs: int) -> list[list[int]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.seed_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]

    def fetch_grad_records(
        self, earliest_record_needs: int
    ) -> list[list[torch.Tensor]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.grad_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]


# TODO Make sure all client model intialized with same weight.
# TODO Support Gradient Pruning
class CeZO_Server:
    def __init__(
        self,
        clients: Sequence[AbstractClient],
        device: torch.device,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.seed_grad_records = SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(len(self.clients))]

        self.server_model: torch.nn.Module | None = None
        self.server_criterion: torch.nn.Module | None = None
        self.optim: torch.optim.Optimizer | None = None

    def set_server_model_and_criterion(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.server_model = model
        self.server_criterion = criterion
        self.optim = optimizer

    def get_sampled_client_index(self):
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def train_one_step(self, iteration: int) -> None:
        # Step 0: initiate something
        sampled_client_index = self.get_sampled_client_index()
        seeds = [random.randint(0, 1e6) for _ in range(self.local_update_steps)]

        # Step 1 & 2: pull model and local update
        local_grad_scalar_list: list[list[torch.Tensor]] = []
        for index in sampled_client_index:
            client = self.clients[index]
            last_update_iter = self.client_last_updates
            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = self.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = self.seed_grad_records.fetch_grad_records(last_update_iter)
            client.pull_model(seeds_list, grad_list)

            local_grad_scalar_list.append(client.local_update(seeds=seeds))
            self.client_last_updates[index] = iteration

        # Step 3: server-side aggregation
        avg_grad_scalar = sum(local_grad_scalar_list).div_(self.num_sample_clients)
        self.seed_grad_records.add_records(seeds=seeds, grad=avg_grad_scalar)

        # Step 4: client sync-to server (older version).
        for index in sampled_client_index:
            client = self.clients[index]
            client.reset_model(seeds=seeds, gradient_scalar=grad)

        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        self.seed_grad_records.remove_too_old(
            earliest_record_needs=min(self.client_last_updates)
        )

        if self.server_model:
            update_model_given_seed_and_grad(
                self.server_model, self.optim, self.device, seeds, avg_grad_scalar
            )

    def eval_model(self, test_loader: Iterable[Any]) -> tuple[float, float]:
        if self.server_model is None:
            raise RuntimeError("set_server_model_and_criterion for server first.")
        self.server_model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_loader):
                if self.device != torch.device("cpu"):
                    images, labels = images.to(self.device), labels.to(self.device)
                pred = self.server_model(images)
                eval_loss.update(self.criterion(pred, labels))
                eval_accuracy.update(accuracy(pred, labels))
        print(
            f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg:.4f}, "
            f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
        )
        return eval_loss.avg, eval_accuracy.avg
