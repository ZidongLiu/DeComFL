from os import path
from typing import Any

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.coordinate_gradient_estimator import CoordinateGradientEstimator as CGE
from cezo_fl.models.cnn_fashion import CNN_FMNIST
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.models.lenet import LeNet
from cezo_fl.models.lstm import CharLSTM
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util import model_helpers
from cezo_fl.util.checkpoint import CheckPoint
from cezo_fl.util.metrics import Metric, accuracy
from config import get_args_str, get_params
from preprocess import preprocess


def prepare_settings(args, device):
    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset == "cifar10":
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=5e-4,
            momentum=args.momentum,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    elif args.dataset == "shakespeare":
        model = CharLSTM().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

    if args.grad_estimate_method in ["rge-central", "rge-forward"]:
        method = args.grad_estimate_method[4:]
        print(f"Using RGE {method}")
        grad_estimator = RGE(
            model,
            parameters=model_helpers.get_trainable_model_parameters(model),
            mu=args.mu,
            num_pert=args.num_pert,
            grad_estimate_method=method,
            device=device,
        )
    elif args.grad_estimate_method in ["cge-forward"]:
        print("Using CGE forward")
        grad_estimator = CGE(
            model,
            mu=args.mu,
            device=device,
        )
    else:
        raise Exception(f"Grad estimate method {args.grad_estimate_method} not supported")
    return model, criterion, optimizer, scheduler, grad_estimator


def get_warmup_lr(args: Any, current_epoch: int, current_iter: int, iters_per_epoch: int) -> float:
    assert isinstance(args.lr, float) and isinstance(args.warmup_epochs, int)
    overall_iterations = args.warmup_epochs * iters_per_epoch + 1
    current_iterations = current_epoch * iters_per_epoch + current_iter + 1
    return args.lr * current_iterations / overall_iterations


def train_model(epoch: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    iter_per_epoch = len(train_loader)
    with tqdm(total=iter_per_epoch, desc="Training:") as t, torch.no_grad():
        for iteration, (images, labels) in enumerate(train_loader):
            if epoch < args.warmup_epochs:
                warmup_lr = get_warmup_lr(args, epoch, iteration, iter_per_epoch)
                for p in optimizer.param_groups:
                    p["lr"] = warmup_lr

            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            # update models
            optimizer.zero_grad()
            grad_estimator.compute_grad(images, labels, criterion, seed=iteration**2 + iteration)
            optimizer.step()

            pred = model(images)
            train_loss.update(criterion(pred, labels))
            train_accuracy.update(accuracy(pred, labels))
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)
        if epoch > args.warmup_epochs:
            scheduler.step()
    return train_loss.avg, train_accuracy.avg


def eval_model(epoch: int) -> tuple[float, float]:
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
    torch.manual_seed(args.seed)

    # set num_clients = 1 to make sure there's 1 train_loader
    args.num_clients = 1
    device_map, train_loaders, test_loader = preprocess(args)
    train_loader = train_loaders[0]
    device = device_map["server"]

    model, criterion, optimizer, scheduler, grad_estimator = prepare_settings(args, device)

    checkpoint = CheckPoint(args, model, optimizer, grad_estimator)

    args_str = get_args_str(args) + "-" + model.model_name
    if args.log_to_tensorboard:
        tensorboard_sub_folder = args_str + "-" + model_helpers.get_current_datetime_str()
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        eval_loss, eval_accuracy = eval_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/test", eval_loss, epoch)
            writer.add_scalar("Accuracy/test", eval_accuracy, epoch)

        if checkpoint.should_update(eval_loss, eval_accuracy, epoch):
            checkpoint.save(
                args_str + "-" + model_helpers.get_current_datetime_str(),
                epoch,
                subfolder=args.log_to_tensorboard,
            )

    if args.log_to_tensorboard:
        writer.close()
