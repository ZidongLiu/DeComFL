import torch
from cezo_fl.client import SyncClient
from models.cnn_mnist import CNN_MNIST
from config import FakeArgs
from preprocess import preprocess
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from torch.optim import SGD
from shared.metrics import accuracy
from copy import deepcopy


# NOTE: this unit test only passes for 1e-6
def test_sync_client_reset():
    args = FakeArgs()

    args.dataset = "mnist"
    args.num_clients = 1
    device, train_loaders, _ = preprocess(args)

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

    sync_client = SyncClient(
        model=model,
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
