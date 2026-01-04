from os import path
from typing import Any

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from cezo_fl.util import model_helpers
from cezo_fl.fl_helpers import get_server_name
from cezo_fl.util.metrics import Metric
from cezo_fl.util.language_utils import LLMBatchInput
from cezo_fl.gradient_estimators.random_gradient_estimator_splitted import (
    RandomGradientEstimatorBatch,
    RandomGradientEstimatorParamwise,
)
from cezo_fl.gradient_estimators.adam_forward import (
    AdamForwardGradientEstimatorBatch,
    AdamForwardGradientEstimatorParamwise,
)
from cezo_fl.gradient_estimators.hybrid_gradient_estimator import (
    HybridGradientEstimatorBatch,
    HybridGradientEstimatorParamwise,
)

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
from experiment_helper.prepare_settings import ModelInferences, MetricPacks


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


def prepare_batch(batch: tuple[Any, Any], device: torch.device) -> tuple[Any, torch.Tensor]:
    """Prepare batch for training/evaluation by handling both image and language model tasks.

    Args:
        batch: Tuple containing either (LLMBatchInput, labels) or (images, labels)
        device: Target device for moving tensors

    Returns:
        Tuple of (batch_input, labels) ready for model inference
    """
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

    return batch_input, labels


def update_model_with_gradient_estimator(
    grad_estimator: Any,
    optimizer: torch.optim.Optimizer,
    batch_input: Any,
    labels: torch.Tensor,
    model: torch.nn.Module,
    model_inferences: ModelInferences,
    metrics: MetricPacks,
    iteration: int,
) -> torch.Tensor:
    """Update model using gradient estimator based on its type.

    Returns the computed gradient scalars.
    """
    seed = iteration**2 + iteration
    if isinstance(
        grad_estimator,
        (
            RandomGradientEstimatorParamwise,
            AdamForwardGradientEstimatorParamwise,
            HybridGradientEstimatorParamwise,
        ),
    ):
        dir_grads = grad_estimator._zo_grad_estimate_paramwise(
            batch_input,
            labels,
            lambda x, y: metrics.train_loss(model_inferences.train_inference(model, x), y),
            seed=seed,
        )
        grad_estimator.update_model_given_seed_and_grad(optimizer, [seed], [dir_grads])
        grad_estimator.update_gradient_estimator_given_seed_and_grad([seed], [dir_grads])
    elif isinstance(
        grad_estimator,
        (
            RandomGradientEstimatorBatch,
            AdamForwardGradientEstimatorBatch,
            HybridGradientEstimatorBatch,
        ),
    ):
        optimizer.zero_grad()
        dir_grads = grad_estimator.compute_grad(
            batch_input,
            labels,
            lambda x, y: metrics.train_loss(model_inferences.train_inference(model, x), y),
            seed=seed,
        )
        optimizer.step()
        grad_estimator.update_gradient_estimator_given_seed_and_grad([seed], [dir_grads])
    else:
        raise ValueError(f"Unsupported gradient estimator: {grad_estimator}")

    return dir_grads


def adjust_learning_rate_and_perturbation(
    args: Setting,
    optimizer: torch.optim.Optimizer,
    grad_estimator: Any,
    iteration: int,
) -> None:
    """Adjust learning rate and perturbation number based on iteration count."""
    if args.adjust_perturb:
        # Determine factor and perturbation multiplier based on iteration
        if iteration == 500:
            factor = 0.8
            pert_multiplier = 2
        elif iteration == 1000:
            factor = 0.5
            pert_multiplier = 4
        elif iteration == 2000:
            factor = 0.3
            pert_multiplier = 8
        else:
            return

        # Adjust learning rates
        if args.lr2 is not None and len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]["lr"] = args.lr * factor
            optimizer.param_groups[1]["lr"] = args.lr2 * factor
        else:
            for p in optimizer.param_groups:
                p["lr"] = args.lr * factor

        # Adjust perturbation number
        grad_estimator.num_pert = args.num_pert * pert_multiplier


def train_by_epoch(
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_estimator: Any,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    model_inferences: ModelInferences,
    metrics: MetricPacks,
    writer: SummaryWriter | None,
) -> None:
    """Train by epoch: process all batches in one epoch."""
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    iter_per_epoch = len(train_loader)

    for epoch in range(args.epoch):
        with tqdm(total=iter_per_epoch, desc="Training:") as t, torch.no_grad():
            for iteration, batch in enumerate(train_loader):
                total_iteration = epoch * iter_per_epoch + iteration

                # Prepare batch for training
                batch_input, labels = prepare_batch(batch, device)

                # update models
                update_model_with_gradient_estimator(
                    grad_estimator,
                    optimizer,
                    batch_input,
                    labels,
                    model,
                    model_inferences,
                    metrics,
                    iteration,
                )

                # Apply learning rate and perturbation adjustments
                adjust_learning_rate_and_perturbation(
                    args, optimizer, grad_estimator, total_iteration
                )

                pred = model_inferences.train_inference(model, batch_input)
                train_loss.update(metrics.train_loss(pred, labels))
                train_accuracy.update(metrics.train_acc(pred, labels))
                t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
                t.update(1)

        # Logging and evaluation
        if args.log_to_tensorboard and writer is not None:
            writer.add_scalar("Loss/train", train_loss.avg, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy.avg, epoch)
        eval_loss, eval_accuracy = eval_model(
            epoch, model, test_loader, device, model_inferences, metrics
        )
        if args.log_to_tensorboard and writer is not None:
            writer.add_scalar("Loss/test", eval_loss, epoch)
            writer.add_scalar("Accuracy/test", eval_accuracy, epoch)


def train_by_iteration(
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_estimator: Any,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    model_inferences: ModelInferences,
    metrics: MetricPacks,
    writer: SummaryWriter | None,
) -> None:
    """Train by iteration: process batches one at a time for a fixed number of iterations."""
    model.train()
    train_loss_metric = Metric("train loss")
    train_accuracy_metric = Metric("train accuracy")
    batch_iter = iter(train_loader)

    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        for iteration in range(args.iterations):
            # Get next batch, cycle through dataloader if exhausted
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)

            # Prepare batch for training
            batch_input, labels = prepare_batch(batch, device)

            # update models
            update_model_with_gradient_estimator(
                grad_estimator,
                optimizer,
                batch_input,
                labels,
                model,
                model_inferences,
                metrics,
                iteration,
            )

            # Apply learning rate and perturbation adjustments
            adjust_learning_rate_and_perturbation(args, optimizer, grad_estimator, iteration)

            pred = model_inferences.train_inference(model, batch_input)
            step_loss = metrics.train_loss(pred, labels)
            step_accuracy = metrics.train_acc(pred, labels)
            # Convert tensors to Python floats if needed
            loss_val = float(step_loss.item() if isinstance(step_loss, torch.Tensor) else step_loss)
            acc_val = float(
                step_accuracy.item() if isinstance(step_accuracy, torch.Tensor) else step_accuracy
            )

            train_loss_metric.update(loss_val)
            train_accuracy_metric.update(acc_val)
            t.set_postfix({"Loss": train_loss_metric.avg, "Accuracy": train_accuracy_metric.avg})
            t.update(1)

            if args.log_to_tensorboard and writer is not None:
                writer.add_scalar("Loss/train", loss_val, iteration)
                writer.add_scalar("Accuracy/train", acc_val, iteration)

            # Evaluation
            if args.eval_iterations != 0 and (iteration + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = eval_model(
                    iteration, model, test_loader, device, model_inferences, metrics
                )
                if args.log_to_tensorboard and writer is not None:
                    writer.add_scalar("Loss/test", eval_loss, iteration)
                    writer.add_scalar("Accuracy/test", eval_accuracy, iteration)


def eval_model(
    epoch: int,
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_inferences: ModelInferences,
    metrics: MetricPacks,
) -> tuple[float, float]:
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # Prepare batch for evaluation
            batch_input, labels = prepare_batch(batch, device)

            pred = model_inferences.test_inference(model, batch_input)
            eval_loss.update(metrics.test_loss(pred, labels))
            eval_accuracy.update(metrics.test_acc(pred, labels))
    print(
        f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg:.4f}, "
        f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


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
        model=model,
        dataset=args.dataset,
        optimizer_setting=args.optimizer_setting,
        rge_setting=args.rge_setting,
    )
    grad_estimator = prepare_settings.get_gradient_estimator(
        model=model, device=device, rge_setting=args.rge_setting, model_setting=args.model_setting
    )

    if args.log_to_tensorboard:
        tensorboard_sub_folder = (
            str(getattr(model, "model_name", "model"))
            + "-"
            + model_helpers.get_current_datetime_str()
        )
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset.value,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )
    else:
        writer = None

    if args.train_by_epoch:
        # Training by epoch
        train_by_epoch(
            args,
            model,
            optimizer,
            grad_estimator,
            train_loader,
            test_loader,
            device,
            model_inferences,
            metrics,
            writer,
        )
    else:
        # Training by iteration/batch (default)
        train_by_iteration(
            args,
            model,
            optimizer,
            grad_estimator,
            train_loader,
            test_loader,
            device,
            model_inferences,
            metrics,
            writer,
        )

    if args.log_to_tensorboard:
        assert writer is not None
        writer.close()
