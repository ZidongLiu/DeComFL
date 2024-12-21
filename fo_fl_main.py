from os import path

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.fl_helpers import get_client_name
from cezo_fl.util import model_helpers
from experiment_helper import prepare_settings
from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
)
from fed_avg.client import FedAvgClient
from fed_avg.server import FedAvgServer
from preprocess import preprocess


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    FederatedLearningSetting,
):
    pass


def setup_server_and_clients(
    args, device_map: dict[str, torch.device], train_loaders
) -> FedAvgServer:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args
    )
    clients = []

    for i in range(args.num_clients):
        client_name = get_client_name(i)
        client_device = device_map[client_name]
        client_model = prepare_settings.get_model(args.dataset, args, args.seed)
        client_model.to(client_device)
        client_optimizer = prepare_settings.get_optimizer(client_model, args.dataset, args)

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

    server_device = device_map["server"]

    server_model = prepare_settings.get_model(args.dataset, args, args.seed)
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
    )

    return server


if __name__ == "__main__":
    args = CliSetting()
    if args.dataset == "shakespeare":
        args.num_clients = 139
    print(args)
    device_map, train_loaders, test_loader = preprocess(args)

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
