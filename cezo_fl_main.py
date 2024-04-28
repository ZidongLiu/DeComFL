from copy import deepcopy
import torch.nn as nn
import torch

from config import get_params
from preprocess import preprocess

from cezo_fl.server import CeZO_Server
from cezo_fl.client import Client

from models.cnn_mnist import CNN_MNIST
from models.lenet import LeNet
from models.cnn_fashion import CNN_FMNIST
from models.lstm import CharLSTM


def prepare_settings(args, device):
    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == "cifar10":
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == "shakespeare":
        model = CharLSTM().to(device)
        criterion = nn.CrossEntropyLoss()

    return model, criterion


# def get_warmup_lr(
#     args, current_epoch: int, current_iter: int, iters_per_epoch: int
# ) -> float:
#     overall_iterations = args.warmup_epochs * iters_per_epoch + 1
#     current_iterations = current_epoch * iters_per_epoch + current_iter + 1
#     return args.lr * current_iterations / overall_iterations


if __name__ == "__main__":
    args = get_params().parse_args()

    device, _, _ = preprocess(args)
    model, criterion = prepare_settings(args, device)
    model.to(device)
    num_of_device = 5
    clients = []

    for i in range(num_of_device):
        _, train_loader, _ = preprocess(args)

        client = Client(
            deepcopy(model),
            train_loader,
            device,
            {
                "method": args.grad_estimate_method,
                "mu": args.mu,
                "num_pert": args.num_pert,
            },
            {"method": "SGD", "lr": args.lr, "momentum": args.momentum},
            criterion,
        )

        clients.append(client)

    server = CeZO_Server(clients, device, num_sample_clients=3, local_update_steps=5)

    with torch.no_grad():
        for ite in range(1000):
            print("train iteration: ", ite)
            server.train_one_step(ite)
            # TODO: train error
            # TODO: eval error
