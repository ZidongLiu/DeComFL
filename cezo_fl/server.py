import abc
import random
import torch
from typing import Sequence
from collections import deque

from dataclasses import dataclass


class AbstractClient:

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> Sequence[torch.Tensor]:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The shape of each tensor should be either a scalar or a num_pertrub*1.
        """
        return NotImplemented

    @abc.abstractmethod
    def sync_to_server(
        self, seeds: Sequence[int], gradient_scalar: Sequence[torch.Tensor]
    ) -> None:
        return NotImplemented

    @abc.abstractmethod
    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        return NotImplemented


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


class Server:
    def __init__(
        self,
        clients: Sequence[AbstractClient],
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.seed_grad_records = SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(len(self.clients))]
        self.server_model = None

    def train_one_step(self, iteration: int) -> None:
        sampled_client_index = random.sample(
            range(len(self.clients)), self.num_sample_clients
        )
        seeds = [random.randint(0, 1e6) for _ in range(self.local_update_steps)]
        local_grad_scalar_list: list[list[torch.Tensor]] = []

        for index in sampled_client_index:
            client = self.clients[index]
            last_update_iter = self.client_last_updates

            seeds_list = self.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = self.seed_grad_records.fetch_grad_records(last_update_iter)
            client.pull_model(seeds_list, grad_list)

            local_grad_scalar_list.append(client.local_update(seeds=seeds))
            self.client_last_updates[index] = iteration

        avg_grad_scalar = sum(local_grad_scalar_list).div_(self.num_sample_clients)
        self.seed_grad_records.add_records(seeds=seeds, grad=avg_grad_scalar)

        for index in sampled_client_index:
            client = self.clients[index]
            # We know the iteration must return only the latest one
            seeds = self.seed_grad_records.fetch_seed_records(iteration)[0]
            grad = self.seed_grad_records.fetch_grad_records(iteration)[0]
            client.sync_to_server(seeds=seeds, gradient_scalar=grad)

        self.seed_grad_records.remove_too_old(
            earliest_record_needs=min(self.client_last_updates)
        )

    def eval_model(self) -> None:
        # TODO
        return NotImplemented
