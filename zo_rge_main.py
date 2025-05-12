from os import path
from typing import Any

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.util import model_helpers
from cezo_fl.fl_helpers import get_server_name
from cezo_fl.util.metrics import Metric, accuracy

from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    OptimizerSetting,
    ModelSetting,
    NormalTrainingLoopSetting,
    RGESetting,
)
from experiment_helper.device import use_device
from experiment_helper.data import (
    get_dataloaders,
    ImageClassificationTask,
    LmClassificationTask,
    LmGenerationTask,
)
from experiment_helper import prepare_settings


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    dataset: ImageClassificationTask | LmClassificationTask | LmGenerationTask,
) -> torch.optim.lr_scheduler.LRScheduler:
    if args.dataset == ImageClassificationTask.mnist:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset in [ImageClassificationTask.cifar10, ImageClassificationTask.fashion]:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    else:
        raise Exception(f"{dataset.value} not support yet in zo_rge_main")


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
            seed = iteration**2 + iteration
            dir_grads = grad_estimator.compute_grad(
                images,
                labels,
                lambda x, y: criterion(model_inferences.train_inference(model, x), y),
                seed=seed,
            )
            grad_estimator.update_gradient_estimator_given_seed_and_grad([seed], [dir_grads])
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


class Setting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    OptimizerSetting,
    ModelSetting,
    RGESetting,
    NormalTrainingLoopSetting,
):
    """
    This is a replacement for regular argparse module.
    We used a third party library pydantic_setting to make command line interface easier to manage.
    Example:
    if __name__ == "__main__":
        args = CliSetting()

    args will have all parameters defined by all components.
    """

    pass


if __name__ == "__main__":
    args = Setting()
    torch.manual_seed(args.seed)

    device_map = use_device(args.device_setting, 1)
    train_loaders, test_loader = get_dataloaders(
        args.data_setting, 1, args.seed, args.get_hf_model_name()
    )
    train_loader = train_loaders[0]
    device = device_map[get_server_name()]

    criterion = torch.nn.CrossEntropyLoss()
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed).to(device)
    optimizer = prepare_settings.get_optimizer(
        model=model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    scheduler = get_scheduler(optimizer, args.dataset)
    grad_estimator = prepare_settings.get_gradient_estimator(
        model=model, device=device, rge_setting=args.rge_setting, model_setting=args.model_setting
    )

    if args.log_to_tensorboard:
        tensorboard_sub_folder = model.model_name + "-" + model_helpers.get_current_datetime_str()
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset.value,
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

    if args.log_to_tensorboard:
        writer.close()
