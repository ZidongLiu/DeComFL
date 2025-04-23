import functools
from os import path
import torch


from tensorboardX import SummaryWriter
from tqdm import tqdm

from byzantine import aggregation as byz_agg
from byzantine import attack as byz_attack
from cezo_fl.client import ResetClient
from cezo_fl.fl_helpers import get_client_name, get_server_name
from cezo_fl.server import CeZO_Server
from cezo_fl.util import model_helpers
from experiment_helper import prepare_settings
from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    RGESetting,
    ByzantineSetting,
)
from experiment_helper.device import use_device
from experiment_helper.data import get_dataloaders


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    RGESetting,
    ByzantineSetting,
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


def setup_server_and_clients(
    args: CliSetting, device_map: dict[str, torch.device], train_loaders
) -> CeZO_Server:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    clients = []

    for i in range(args.num_clients):
        client_name = get_client_name(i)
        client_device = device_map[client_name]
        client_model = prepare_settings.get_model(
            dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
        ).to(client_device)
        client_optimizer = prepare_settings.get_optimizer(
            model=client_model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
        )
        client_grad_estimator = prepare_settings.get_gradient_estimator(
            model=client_model,
            device=client_device,
            rge_setting=args.rge_setting,
            model_setting=args.model_setting,
        )

        client = ResetClient(
            client_model,
            model_inferences.train_inference,
            train_loaders[i],
            client_grad_estimator,
            client_optimizer,
            metrics.train_loss,
            metrics.train_acc,
            client_device,
        )
        clients.append(client)

    server_device = device_map[get_server_name()]
    server = CeZO_Server(
        clients,
        server_device,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update_steps,
    )

    # set server tools
    server_model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(server_device)
    server_optimizer = prepare_settings.get_optimizer(
        model=server_model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    server_grad_estimator = prepare_settings.get_gradient_estimator(
        model=server_model,
        device=server_device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
    )

    server.set_server_model_and_criterion(
        server_model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        server_optimizer,
        server_grad_estimator,
    )

    # TODO(lizhe) move this into a seperate main file.
    # Prepare the Byzantine attack
    if args.byz_type == "no_byz":
        server.register_attack_func(byz_attack.no_byz)
    elif args.byz_type == "gaussian":
        server.register_attack_func(
            functools.partial(byz_attack.gaussian_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "sign":
        server.register_attack_func(
            functools.partial(byz_attack.sign_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "trim":
        server.register_attack_func(
            functools.partial(byz_attack.trim_attack, num_attack=args.num_byz)
        )
    elif args.byz_type == "krum":
        server.register_attack_func(
            functools.partial(byz_attack.krum_attack, f=args.num_byz, lr=args.lr)
        )
    else:
        raise Exception(
            "byz_type should be one of no_byz, gaussian, sign, trim, krum."
            + f"But get {args.byz_type}"
        )

    if args.aggregation == "mean":
        server.register_aggregation_func(byz_agg.mean)
    elif args.aggregation == "median":
        server.register_aggregation_func(byz_agg.median)
    elif args.aggregation == "trim":
        server.register_aggregation_func(byz_agg.trim)
    elif args.aggregation == "krum":
        server.register_aggregation_func(byz_agg.krum)
    else:
        raise Exception(
            "aggregation type should be one of mean, median, trim, krum. "
            + f"But get {args.aggregation}"
        )

    return server


if __name__ == "__main__":
    args = CliSetting()
    print(args)
    device_map = use_device(args.device_setting, args.num_clients)
    train_loaders, test_loader = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )
    server = setup_server_and_clients(args, device_map, train_loaders)

    if args.log_to_tensorboard:
        assert server.server_model
        tensorboard_sub_folder = "-".join(
            [
                server.server_model.model_name,
                model_helpers.get_current_datetime_str(),
            ]
        )
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                "decomfl",
                args.dataset.value,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step(ite)
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            t.update(1)
            if args.adjust_perturb:
                if ite == 500:
                    server.set_learning_rate(args.lr * 0.8)
                    server.set_perturbation(args.num_pert * 2)
                elif ite == 1000:
                    server.set_learning_rate(args.lr * 0.5)
                    server.set_perturbation(args.num_pert * 4)
                elif ite == 2000:
                    server.set_learning_rate(args.lr * 0.3)
                    server.set_perturbation(args.num_pert * 8)

            if args.log_to_tensorboard:
                writer.add_scalar("Loss/train", step_loss, ite)
                writer.add_scalar("Accuracy/train", step_accuracy, ite)
            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader)
                if args.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Accuracy/test", eval_accuracy, ite)
