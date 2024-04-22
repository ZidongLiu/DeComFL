import torch
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy
from pruning.helpers import generate_random_mask_arr
from config import get_params
from preprocess import preprocess, use_sparsity_dict
from models.cnn_mnist import CNN_MNIST
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from models.resnet_cifar10 import resnet20
from models.lenet import LeNet
from models.cnn_fashion import CNN_FMNIST


def prepare_settings(args, device):
    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
        model_name = "CNN_MNIST"
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset == "cifar10":
        model = LeNet().to(device)
        model_name = "LeNet"
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=args.momentum
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200], gamma=0.1
        )
    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(device)
        model_name = "CNN_FMNIST"
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=args.momentum
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10000], gamma=0.1
        )

    rge = RGE(
        model,
        mu=args.mu,
        num_pert=args.num_pert,
        grad_estimate_method=args.grad_estimate_method,
        device=device,
    )
    return model, criterion, optimizer, scheduler, rge, model_name


def get_warmup_lr(
    args, current_epoch: int, current_iter: int, iters_per_epoch: int
) -> float:
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
            rge.compute_grad(images, labels, criterion)
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

    if args.log_to_tensorboard:
        tensorboard_sub_folder = (
            f"rge-{args.grad_estimate_method}-{args.log_to_tensorboard}-"
            + f"num_pert-{args.num_pert}-{get_current_datetime_str()}"
        )
        writer = SummaryWriter(
            path.join("tensorboards", args.dataset, tensorboard_sub_folder)
        )

    device, train_loader, test_loader = preprocess(args)
    model, criterion, optimizer, scheduler, rge, model_name = prepare_settings(
        args, device
    )

    sparsity_dict = use_sparsity_dict(args, model_name)
    for epoch in range(args.epoch):
        if sparsity_dict is not None and epoch % args.mask_shuffle_interval == 0:
            print("Updating gradient mask!")
            mask_arr = generate_random_mask_arr(model, sparsity_dict, device)
            rge.set_prune_mask_arr(mask_arr)

        train_loss, train_accuracy = train_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        eval_loss, eval_accuracy = eval_model(epoch)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/test", eval_loss, epoch)
            writer.add_scalar("Accuracy/test", eval_accuracy, epoch)

    if args.log_to_tensorboard:
        writer.close()
