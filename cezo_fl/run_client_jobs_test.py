from unittest.mock import MagicMock
import pytest
import torch


from cezo_fl.client import SyncClient
from cezo_fl.run_client_jobs import parallalizable_client_job, execute_sampled_clients
from models.cnn_mnist import CNN_MNIST
from config import FakeArgs
from preprocess import preprocess
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from torch.optim import SGD
from shared.metrics import accuracy
from copy import deepcopy


def set_fake_clients() -> list[SyncClient]:
    args = FakeArgs()
    args.dataset = "mnist"
    args.num_clients = 3
    args.num_pert = 4
    args.local_update_steps = 2

    device_map, train_loaders, _ = preprocess(args)
    device = device_map["server"]
    fake_clients = []
    for i in range(args.num_clients):
        torch.random.manual_seed(1234)  # Make sure all models are the same
        model = CNN_MNIST().to(device)
        train_loader = train_loaders[0]
        grad_estimator = RGE(
            model,
            mu=1e-3,
            num_pert=2,
            grad_estimate_method="forward",
            device=device,
        )
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0)
        criterion = torch.nn.CrossEntropyLoss()
        fake_clients.append(
            SyncClient(
                model=model,
                dataloader=train_loader,
                grad_estimator=grad_estimator,
                optimizer=optimizer,
                criterion=criterion,
                accuracy_func=accuracy,
                device=device,
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


def test_execute_sampled_clients_parallabel():
    server = MagicMock()
    server.device = "cpu"
    server.client_last_updates = [0, 0, 0]
    server.seed_grad_records.fetch_seed_records.return_value = [[1, 2], [1, 2], [1, 2]]
    server.seed_grad_records.fetch_grad_records.return_value = [
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
        [torch.tensor([1, 1, 1, 1]), torch.tensor([-1, -1, -1, -1])],
    ]

    for _ in range(10):  # Try multiple time
        server.clients = set_fake_clients()
        serialized_result = execute_sampled_clients(
            server, sampled_client_index=[0, 1, 2], seeds=[7, 8], parallel=False
        )
        server.clients = set_fake_clients()  # Reset client
        parallel_result = execute_sampled_clients(
            server, sampled_client_index=[0, 1, 2], seeds=[7, 8], parallel=True
        )
        # result is (step_train_loss, step_train_accuracy, local_grad_scalar_list)
        assert abs(serialized_result[0].avg - parallel_result[0].avg) < 1e-5
        assert abs(serialized_result[1].avg - parallel_result[1].avg) < 1e-5

        for s_local_grad, p_local_grad in zip(serialized_result[2], parallel_result[2]):
            for s_local_grad_one_step, p_local_grad_one_step in zip(s_local_grad, p_local_grad):
                assert (s_local_grad_one_step - p_local_grad_one_step).abs().max() < 1e-5
