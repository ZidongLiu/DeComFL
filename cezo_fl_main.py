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

from shared.metrics import Metric, accuracy


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


def eval_model(client, epoch: int) -> tuple[float, float]:
    model = client.model
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            pred = model(images)
            eval_loss.update(criterion(pred, labels))
            eval_accuracy.update(accuracy(pred, labels))
    print(
        f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg:.4f}, "
        f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


if __name__ == "__main__":
    args = get_params().parse_args()

    device, _, test_loader = preprocess(args)
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
            if (ite + 1) % 20 == 0:
                eval_model(clients[0], ite)
