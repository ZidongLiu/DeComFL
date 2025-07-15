from os import path

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.fl_helpers import get_client_name, get_server_name
from cezo_fl.util import model_helpers
from experiment_helper import prepare_settings
from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    FOFLSetting,
)
from fed_avg.client import FedAvgClient
from fed_avg.server import FedAvgServer
from experiment_helper.data import get_dataloaders
from experiment_helper.device import use_device


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
    FOFLSetting,
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
    args, device_map: dict[str, torch.device], train_loaders
) -> FedAvgServer:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    clients = []

    for i in range(args.num_clients):
        client_name = get_client_name(i)
        client_device = device_map[client_name]
        client_model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed)
        client_model.to(client_device)
        client_optimizer = prepare_settings.get_optimizer(
            client_model, args.dataset, args.optimizer_setting
        )
        if not isinstance(client_optimizer, torch.optim.SGD):
            raise ValueError("Only SGD optimizer is supported for FO-FL")

        client = FedAvgClient(
            client_model,
            model_inferences.train_inference,
            train_loaders[i],
            client_optimizer,
            metrics.train_loss,
            metrics.train_acc,
            client_device,
        )
        clients.append(client)

    server_device = device_map[get_server_name()]

    server_model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed)
    server_model.to(server_device)

    server = FedAvgServer(
        clients,
        server_device,
        server_model=server_model,
        server_model_inference=model_inferences.test_inference,
        server_criterion=metrics.test_loss,
        server_accuracy_func=metrics.test_acc,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_update_steps,
        fo_fl_strategy=args.fo_fl_strategy,
        fo_fl_beta1=args.fo_fl_beta1,
        fo_fl_beta2=args.fo_fl_beta2,
        lr=args.lr,
    )

    return server


if __name__ == "__main__":
    args = CliSetting()
    print(args.fo_fl_strategy)
    device_map = use_device(args.device_setting, args.num_clients)
    train_loaders, test_loader = get_dataloaders(
        args.data_setting, args.num_clients, args.seed, args.get_hf_model_name()
    )

    server = setup_server_and_clients(args, device_map, train_loaders)

    if args.log_to_tensorboard:
        tensorboard_sub_folder = "-".join(
            [
                server.server_model.model_name,
                model_helpers.get_current_datetime_str(),
            ]
        )
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                "fed_avg",
                args.dataset.value,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    with tqdm(total=args.iterations, desc="Training:") as t:
        for ite in range(args.iterations):
            step_loss, step_accuracy = server.train_one_step()
            t.set_postfix({"Loss": step_loss, "Accuracy": step_accuracy})
            t.update(1)

            if args.log_to_tensorboard:
                writer.add_scalar("Loss/train", step_loss, ite)
                writer.add_scalar("Accuracy/train", step_accuracy, ite)
            # eval
            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = server.eval_model(test_loader, ite)
                if args.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Accuracy/test", eval_accuracy, ite)
