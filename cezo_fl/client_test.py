from copy import deepcopy

import torch
from torch.optim import SGD

from cezo_fl.client import SyncClient
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util.metrics import accuracy
from config import FakeArgs
from preprocess import preprocess


# NOTE: this unit test only passes for 1e-6
def test_sync_client_reset():
    args = FakeArgs()

    args.dataset = "mnist"
    args.num_clients = 1
    device_map, train_loaders, _ = preprocess(args)
    device = device_map["server"]

    model = CNN_MNIST().to(device)

    train_loader = train_loaders[0]
    grad_estimator = RGE(
        model.parameters(),
        mu=1e-3,
        num_pert=2,
        grad_estimate_method="rge-forward",
        device=device,
    )
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0)

    criterion = torch.nn.CrossEntropyLoss()

    sync_client = SyncClient(
        model=model,
        model_inference=lambda m, x: m(x),
        dataloader=train_loader,
        grad_estimator=grad_estimator,
        optimizer=optimizer,
        criterion=criterion,
        accuracy_func=accuracy,
        device=device,
    )

    original_model = deepcopy(model)
    with torch.no_grad():
        sync_client.local_update([1, 2, 3, 4, 5])
        sync_client.local_update([6, 7, 8, 9, 10])
        sync_client.reset_model()

    for orig_param, reset_param in zip(original_model.parameters(), model.parameters()):
        assert (orig_param - reset_param).abs().max() < 1e-6
