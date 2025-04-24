from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD

from cezo_fl.client import ResetClient
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.gradient_estimators.random_gradient_estimator import (
    RandomGradEstimateMethod,
    RandomGradientEstimator,
)
from cezo_fl.run_client_jobs import execute_sampled_clients, parallalizable_client_job
from cezo_fl.util.metrics import accuracy
from cezo_fl.fl_helpers import get_server_name
from experiment_helper import device, cli_parser


def get_mnist_data_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    return torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)


class Setting(cli_parser.OptimizerSetting, cli_parser.DeviceSetting, cli_parse_args=False):
    pass


def set_fake_clients(
    num_clients: int = 3, num_pert: int = 4, local_update_steps: int = 2
) -> list[ResetClient]:
    args = Setting()
    device_map = device.use_device(args.device_setting, num_clients=num_clients)
    model_device = device_map[get_server_name()]
    fake_clients = []
    assert isinstance(args.lr, float)
    for i in range(num_clients):
        torch.random.manual_seed(1234)  # Make sure all models are the same
        model = CNN_MNIST().to(model_device)
        train_loader = get_mnist_data_loader()
        grad_estimator = RandomGradientEstimator(
            model.parameters(),
            mu=1e-3,
            num_pert=2,
            grad_estimate_method=RandomGradEstimateMethod.rge_forward,
            device=model_device,
        )
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0)
        criterion = torch.nn.CrossEntropyLoss()
        fake_clients.append(
            ResetClient(
                model=model,
                model_inference=lambda m, x: m(x),
                dataloader=train_loader,
                grad_estimator=grad_estimator,
                optimizer=optimizer,
                criterion=criterion,
                accuracy_func=accuracy,
                device=model_device,
            )
        )
    return fake_clients


def test_parallalizable_client_job_identical():
    fake_clients = set_fake_clients()
    # Note in the fake_client setup, we choose local_update=2, clients=3, and num_pert=4
    # We need to make sure each job runs on are completely independent.
    pull_seeds_list = [[1, 2], [1, 2], [1, 2]]
    pull_grad_list = [
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
    ]
    results = []
    for fake_client in fake_clients:
        results.append(
            parallalizable_client_job(
                fake_client,
                pull_seeds_list,
                pull_grad_list,
                local_update_seeds=[7, 8],
                server_device=torch.device("cpu"),
            )
        )
        print(results[-1])
    # Because we give the same model, same seed and grad scalar,  the local update must be the same.
    for i in range(2):  # local_update
        assert (results[0].grad_tensors[i] - results[1].grad_tensors[i]).abs().max() < 1e-6
        assert (results[1].grad_tensors[i] - results[2].grad_tensors[i]).abs().max() < 1e-6

    assert abs(results[0].step_accuracy - results[1].step_accuracy) < 1e-6
    assert abs(results[1].step_accuracy - results[2].step_accuracy) < 1e-6

    assert abs(results[0].step_loss - results[1].step_loss) < 1e-6
    assert abs(results[1].step_loss - results[2].step_loss) < 1e-6


@pytest.mark.parametrize("num_clients", [1, 3, 5])
@pytest.mark.parametrize("num_pert", [1, 3, 5])
@pytest.mark.parametrize("local_update_steps", [1, 3, 5])
def test_execute_sampled_clients_parallabel(num_clients, num_pert, local_update_steps):
    server = MagicMock()
    server.device = torch.device("cpu")
    server.client_last_updates = [0 for _ in range(num_clients)]
    existing_iteration = 3
    server.seed_grad_records.fetch_seed_records.return_value = np.random.randint(
        0, 100, (existing_iteration, local_update_steps)
    ).tolist()
    server.seed_grad_records.fetch_grad_records.return_value = [
        [torch.randn(num_pert) for _ in range(local_update_steps)]
        for _ in range(existing_iteration)
    ]

    for _ in range(3):  # Try multiple time
        server.clients = set_fake_clients(num_clients, num_pert, local_update_steps)

        sampled_index = np.random.choice(
            [i for i in range(num_clients)], (num_clients + 1) // 2, replace=False
        ).tolist()
        seeds = np.random.randint(1, 100, local_update_steps).tolist()

        serialized_result = execute_sampled_clients(
            server, sampled_client_index=sampled_index, seeds=seeds, parallel=False
        )
        server.clients = set_fake_clients(num_clients, num_pert, local_update_steps)  # Reset client
        parallel_result = execute_sampled_clients(
            server, sampled_client_index=sampled_index, seeds=seeds, parallel=True
        )
        # result is (step_train_loss, step_train_accuracy, local_grad_scalar_list)
        assert abs(serialized_result[0].avg - parallel_result[0].avg) < 1e-4
        assert abs(serialized_result[1].avg - parallel_result[1].avg) < 1e-4

        for s_local_grad, p_local_grad in zip(serialized_result[2], parallel_result[2]):
            for s_local_grad_one_step, p_local_grad_one_step in zip(s_local_grad, p_local_grad):
                assert (s_local_grad_one_step - p_local_grad_one_step).abs().max() < 1e-4
