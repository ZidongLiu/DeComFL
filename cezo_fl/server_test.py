from cezo_fl.server import (
    AbstractClient,
    CeZO_Server,
    SeedAndGradientRecords,
    update_model_given_seed_and_grad,
)
from typing import Sequence
from unittest.mock import MagicMock, patch

import torch
import pytest


def test_seed_records():
    sr = SeedAndGradientRecords()

    sr.add_records([1, 2, 3], [None] * 3)  # iter 0
    sr.add_records([2, 3, 4], [None] * 3)  # iter 1
    assert sr.fetch_seed_records(earliest_record_needs=0) == [[1, 2, 3], [2, 3, 4]]

    sr.add_records([3, 4, 5], [None] * 3)  # iter 2
    sr.remove_too_old(earliest_record_needs=1)
    assert sr.fetch_seed_records(earliest_record_needs=2) == [[3, 4, 5]]
    assert sr.fetch_seed_records(earliest_record_needs=3) == []


def test_update_model_given_seed_and_grad():
    # Make the update second times and the output suppose to be the same.
    ouputs = []
    for _ in range(2):
        torch.manual_seed(0)
        fake_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim = torch.optim.SGD(fake_model.parameters(), lr=1e-3)
        updated_model = update_model_given_seed_and_grad(
            fake_model,
            optim,
            seeds=[1, 2, 3],
            grad_scalar_list=[  # two perturbations
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3, 0.4]),
                torch.tensor([0.5, 0.6]),
            ],
        )
        ouputs.append(
            updated_model(
                torch.tensor(
                    [list(range(i, 10 + i)) for i in range(3)], dtype=torch.float
                )
            )
        )
    assert ouputs[0].shape == (3, 2)
    assert ouputs[1].shape == (3, 2)
    torch.testing.assert_close(ouputs[0], ouputs[1])


class FakeClient(AbstractClient):
    def local_update(self, seeds: Sequence[int]) -> Sequence[torch.Tensor]:
        return [torch.tensor([0.1, 0.2, 0.3]) for _ in range(len(seeds))]

    def reset_model(self) -> None:
        return

    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        return


@patch.object(CeZO_Server, "get_sampled_client_index", return_value=[0, 1])
def test_server_train_one_step(mocke_get_sampled_client_index):
    clients = [FakeClient() for _ in range(4)]
    for client in clients:
        client.pull_model = MagicMock()

    server = CeZO_Server(
        clients=clients,
        device=torch.device("cpu"),
        num_sample_clients=2,
        local_update_steps=3,
    )
    # Just make sure it can execute
    server.train_one_step(0)
    server.train_one_step(1)
    server.train_one_step(2)
