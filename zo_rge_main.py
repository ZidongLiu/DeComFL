from os import path

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.util import model_helpers
from cezo_fl.fl_helpers import get_server_name
from cezo_fl.util.metrics import Metric
from cezo_fl.util.language_utils import LLMBatchInput

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
from experiment_helper.data import get_dataloaders
from experiment_helper import prepare_settings


def train_model(epoch: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    iter_per_epoch = len(train_loader)
    with tqdm(total=iter_per_epoch, desc="Training:") as t, torch.no_grad():
        for iteration, batch in enumerate(train_loader):
            total_iteration = epoch * iter_per_epoch + iteration

            # Handle both image classification and language model tasks
            if isinstance(batch[0], LLMBatchInput):
                # Language model task: batch is (LLMBatchInput, labels)
                batch_input, labels = batch
                if device != torch.device("cpu"):
                    batch_input = batch_input.to(device)
                    labels = labels.to(device)
            else:
                # Image classification task: batch is (images, labels)
                images, labels = batch
                if device != torch.device("cpu"):
                    images, labels = images.to(device), labels.to(device)
                batch_input = images

            # update models
            optimizer.zero_grad()
            seed = iteration**2 + iteration
            dir_grads = grad_estimator.compute_grad(
                batch_input,
                labels,
                lambda x, y: metrics.train_loss(model_inferences.train_inference(model, x), y),
                seed=seed,
            )
            grad_estimator.update_gradient_estimator_given_seed_and_grad([seed], [dir_grads])
            optimizer.step()

            # Apply learning rate and perturbation adjustments
            if args.adjust_perturb:
                if total_iteration == 500:
                    for p in optimizer.param_groups:
                        p["lr"] = args.lr * 0.8
                    grad_estimator.num_pert = args.num_pert * 2
                elif total_iteration == 1000:
                    for p in optimizer.param_groups:
                        p["lr"] = args.lr * 0.5
                    grad_estimator.num_pert = args.num_pert * 4
                elif total_iteration == 2000:
                    for p in optimizer.param_groups:
                        p["lr"] = args.lr * 0.3
                    grad_estimator.num_pert = args.num_pert * 8

            pred = model_inferences.train_inference(model, batch_input)
            train_loss.update(metrics.train_loss(pred, labels))
            train_accuracy.update(metrics.train_acc(pred, labels))
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)
    return train_loss.avg, train_accuracy.avg


def eval_model(epoch: int) -> tuple[float, float]:
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # Handle both image classification and language model tasks
            if isinstance(batch[0], LLMBatchInput):
                # Language model task: batch is (LLMBatchInput, labels)
                batch_input, labels = batch
                if device != torch.device("cpu"):
                    batch_input = batch_input.to(device)
                    labels = labels.to(device)
            else:
                # Image classification task: batch is (images, labels)
                images, labels = batch
                if device != torch.device("cpu"):
                    images, labels = images.to(device), labels.to(device)
                batch_input = images

            pred = model_inferences.test_inference(model, batch_input)
            eval_loss.update(metrics.test_loss(pred, labels))
            eval_accuracy.update(metrics.test_acc(pred, labels))
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

    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed).to(device)
    optimizer = prepare_settings.get_optimizer(
        model=model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
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
