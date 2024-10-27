from typing import Sequence
from unittest.mock import MagicMock, patch

import torch

from cezo_fl.client import AbstractClient, LocalUpdateResult
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.server import CeZO_Server, SeedAndGradientRecords
from cezo_fl.shared import update_model_given_seed_and_grad


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
        update_model_given_seed_and_grad(
            optim,
            RGE(fake_model, num_pert=2),
            iteration_seeds=[1, 2, 3],
            iteration_grad_scalar=[  # two perturbations
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3, 0.4]),
                torch.tensor([0.5, 0.6]),
            ],
        )
        ouputs.append(
            fake_model(torch.tensor([list(range(i, 10 + i)) for i in range(3)], dtype=torch.float))
        )
    assert ouputs[0].shape == (3, 2)
    assert ouputs[1].shape == (3, 2)
    torch.testing.assert_close(ouputs[0], ouputs[1])


class FakeClient(AbstractClient):
    def __init__(self):
        self.device = torch.device("cpu")

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        return LocalUpdateResult(
            grad_tensors=[torch.tensor([0.1, 0.2, 0.3]) for _ in range(len(seeds))],
            step_loss=0.1,
            step_accuracy=0.1,
        )

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
    clients = [FakeClient() for _ in range(3)]
    for client in clients:
        client.pull_model = MagicMock()

    server = CeZO_Server(
        clients=clients,
        device=torch.device("cpu"),
        num_sample_clients=2,
        local_update_steps=3,
    )
    # Mock the execution
    server.train_one_step(0)
    assert server.seed_grad_records.current_iteration == 0
    assert server.seed_grad_records.earliest_records == 0
    assert server.client_last_updates == [0, 0, 0]

    mocke_get_sampled_client_index.return_value = [1, 2]
    server.train_one_step(1)
    assert server.seed_grad_records.current_iteration == 1
    assert server.seed_grad_records.earliest_records == 0
    assert server.client_last_updates == [0, 1, 1]

    mocke_get_sampled_client_index.return_value = [0, 1]
    server.train_one_step(2)
    assert server.seed_grad_records.current_iteration == 2
    assert server.seed_grad_records.earliest_records == 1
    assert server.client_last_updates == [2, 2, 1]

    mocke_get_sampled_client_index.return_value = [1, 2]
    server.train_one_step(3)
    assert server.seed_grad_records.current_iteration == 3
    assert server.seed_grad_records.earliest_records == 2
    assert server.client_last_updates == [2, 3, 3]

    assert len(clients[2].pull_model.call_args_list) == 2  # client 2 is called twice.
    first_pull_model_args = clients[2].pull_model.call_args_list[0][0]
    second_pull_model_args = clients[2].pull_model.call_args_list[1][0]
    assert len(first_pull_model_args[0]) == 1  # Pull the 0-th round seeds.
    assert len(second_pull_model_args[0]) == 2  # Pull the 1-st and 2-nd round seeds.
