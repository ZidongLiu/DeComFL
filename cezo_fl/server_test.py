from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchvision
import torchvision.transforms as transforms

from cezo_fl.client import AbstractClient, LocalUpdateResult, ResetClient
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator,
    RandomGradEstimateMethod,
)
from cezo_fl.gradient_estimators.adam_forward import (
    AdamForwardGradientEstimator,
    KUpdateStrategy,
)
from cezo_fl.server import CeZO_Server, SeedAndGradientRecords
from cezo_fl.util.metrics import accuracy


def get_mnist_data_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)


def test_seed_records():
    sr = SeedAndGradientRecords()

    sr.add_records([1, 2, 3], [None] * 3)  # iter 0
    sr.add_records([2, 3, 4], [None] * 3)  # iter 1
    assert sr.fetch_seed_records(earliest_record_needs=0) == [[1, 2, 3], [2, 3, 4]]

    sr.add_records([3, 4, 5], [None] * 3)  # iter 2
    sr.remove_too_old(earliest_record_needs=1)
    assert sr.fetch_seed_records(earliest_record_needs=2) == [[3, 4, 5]]
    assert sr.fetch_seed_records(earliest_record_needs=3) == []


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


@pytest.mark.parametrize("estimator_type", ["vanilla", "adam_forward"])
@pytest.mark.parametrize(
    "k_update_strategy", [KUpdateStrategy.LAST_LOCAL_UPDATE, KUpdateStrategy.ALL_LOCAL_UPDATES]
)
def test_server_client_model_sync(estimator_type, k_update_strategy):
    # Skip k_update_strategy test for vanilla estimator
    if estimator_type == "vanilla" and k_update_strategy != KUpdateStrategy.LAST_LOCAL_UPDATE:
        pytest.skip("k_update_strategy only applies to adam_forward estimator")

    # Setup test environment
    num_clients = 3
    local_update_steps = 2
    device = torch.device("cpu")
    lr = 1e-4

    # Create clients with same initial model
    clients = []
    for _ in range(num_clients):
        # have to set seed to make the model initialized the same
        torch.manual_seed(0)
        model = CNN_MNIST().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        if estimator_type == "vanilla":
            grad_estimator = RandomGradientEstimator(
                model.parameters(),
                mu=1e-3,
                num_pert=2,
                grad_estimate_method=RandomGradEstimateMethod.rge_forward,
                device=device,
            )
        else:  # adam_forward
            grad_estimator = AdamForwardGradientEstimator(
                model.parameters(),
                mu=1e-3,
                num_pert=2,
                k_update_strategy=k_update_strategy,
                hessian_smooth=0.95,
                device=device,
            )

        client = ResetClient(
            model=model,
            model_inference=lambda m, x: m(x),
            dataloader=get_mnist_data_loader(),
            grad_estimator=grad_estimator,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            accuracy_func=accuracy,
            device=device,
        )
        clients.append(client)

    # Create server with same initial model
    server = CeZO_Server(
        clients=clients,
        device=device,
        num_sample_clients=1,
        local_update_steps=local_update_steps,
    )

    # Set server model and tools
    torch.manual_seed(0)
    server_model = CNN_MNIST().to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr)

    if estimator_type == "vanilla":
        server_grad_estimator = RandomGradientEstimator(
            server_model.parameters(),
            mu=1e-3,
            num_pert=2,
            grad_estimate_method=RandomGradEstimateMethod.rge_forward,
            device=device,
        )
    else:  # adam_forward
        server_grad_estimator = AdamForwardGradientEstimator(
            server_model.parameters(),
            mu=1e-3,
            num_pert=2,
            k_update_strategy=k_update_strategy,
            hessian_smooth=0.95,
            device=device,
        )

    server.set_server_model_and_criterion(
        server_model,
        lambda m, x: m(x),
        torch.nn.CrossEntropyLoss(),
        accuracy,
        server_optimizer,
        server_grad_estimator,
    )

    with torch.no_grad():
        # Run a few training steps
        for i in range(5):
            server.train_one_step(i)

        # for each client try to pull the model from server and compare the model with server's model
        for client_index, client in enumerate(clients):
            # step 1: map pull_grad_list data to client's device
            last_update_iter = server.client_last_updates[client_index]
            pull_grad_list = server.seed_grad_records.fetch_grad_records(last_update_iter)
            pull_seeds_list = server.seed_grad_records.fetch_seed_records(last_update_iter)
            transfered_grad_list = [
                [tensor.to(client.device) for tensor in tensors] for tensors in pull_grad_list
            ]

            # step 2: client pull to update its model to latest
            client.pull_model(pull_seeds_list, transfered_grad_list)

            for server_param, client_param in zip(
                server.server_model.parameters(), client.model.parameters()
            ):
                # Models should be synchronized after each step
                assert (
                    (server_param - client_param).abs().max() < 1e-6
                ), f"Server and client {client_index} model parameters differ for {estimator_type} with {k_update_strategy}"

            if estimator_type == "adam_forward":
                assert isinstance(client.grad_estimator, AdamForwardGradientEstimator)
                # K_vec should be synchronized between server and clients
                assert (
                    (server_grad_estimator.K_vec - client.grad_estimator.K_vec).abs().max() < 1e-6
                ), f"K_vec not synchronized between server and client after step {i}"
