import torch
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy
from config import get_params
from preprocess import preprocess
from models.cnn_mnist import CNN_MNIST
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from models.cnn_cifar10 import CNN_CIFAR10
from models.resnet import ResNet18


def prepare_settings(args, device):
    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset == "cifar10":
        model = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    rge = RGE(
        model,
        mu=args.mu,
        num_pert=args.num_pert,
        grad_estimate_method=args.grad_estimate_method,
        device=device,
    )
    return model, criterion, optimizer, scheduler, rge


def train_model(epoch: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    with tqdm(total=len(train_loader), desc="Training:") as t, torch.no_grad():
        for _, (images, labels) in enumerate(train_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            # update models
            optimizer.zero_grad()
            rge.compute_grad(images, labels, criterion)
            optimizer.step()

            pred = model(images)
            train_loss.update(criterion(pred, labels))
            train_accuracy.update(accuracy(pred, labels))
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)
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

    if args.log_to_tensorboard:
        tensorboard_sub_folder = (
            f"rge-{args.grad_estimate_method}-{args.log_to_tensorboard}-"
            + f"num_pert-{args.num_pert}-{get_current_datetime_str()}"
        )
        writer = SummaryWriter(
            path.join("tensorboards", args.dataset, tensorboard_sub_folder)
        )

    device, train_loader, test_loader = preprocess(args)
    model, criterion, optimizer, scheduler, rge = prepare_settings(args, device)

    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        eval_loss, eval_accuracy = eval_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/test", eval_loss, epoch)
            writer.add_scalar("Accuracy/test", eval_accuracy, epoch)

    writer.close()
